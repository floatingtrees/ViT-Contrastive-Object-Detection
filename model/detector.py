import torch
from torch import nn
from RPN import RPN
from ViT import feature_extraction_ViT, classification_ViT
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import torchvision
from torchvision.transforms import v2
import multiprocessing

def probabilistic_round(tensor):
    # Get the fractional part of each element
    fractional_part = tensor - tensor.floor()
    
    # Generate random numbers in the same shape as the tensor
    random_values = torch.rand_like(tensor)
    
    # Compare random values with the fractional part
    rounded_tensor = torch.where(random_values < fractional_part, tensor.ceil(), tensor.floor())
    
    return rounded_tensor.int()



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class Detector(nn.Module):
    def __init__(self, image_size, # size of input images
                patch_size, # size of each patch
                dim, # last dimension after ViT transformation
                depth, # number of transformer blocks
                heads, # number of heads per attention layer
                mlp_dim, # dimensionality of the feedforward layers within transformer blocks
                image_channels = 3, # number of channels in the image
                encoding_dimensionality = 512, # dimensionality of the class encodings
                classifier_depth = 2, # depth of the classifiers on regression boxes
                classifier_heads = 4, # number of heads on the regression box classifiers
                classifier_mlp_dim = 2048, # mlp dim of the classifier
                classifier_patch_size = 8, # classifier patch size, with respect to the feature map
                box_heights = (2, 4, 8), #possible anchor box heights for RPN, scales with patch size
                box_widths = (2, 4,8), # possible anchor box widths for RPN
                classifier_cores = None
                ):
        super(Detector, self).__init__()
        self.classifier_cores = classifier_cores

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.image_dim = (image_channels, image_height, image_width)
        self.patch_height = patch_height
        self.patch_width = patch_width

        classifier_patch_height, classifier_patch_width = pair(classifier_patch_size)
        feature_map_shape = (dim, image_height// patch_height, image_width // patch_width)
        
        self.feature_extraction_ViT = feature_extraction_ViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
            heads = heads, mlp_dim = mlp_dim, channels=image_channels)
        
       

        self.RPN = RPN(feature_map_shape, projection_dim=dim)
        


        self.anchor_boxes = self.RPN.anchor_boxes
        self.anchor_boxes_height = []
        self.anchor_boxes_width = []
        for element in self.anchor_boxes:
            self.anchor_boxes_height.append(element[0])
            self.anchor_boxes_width.append(element[1])
        self.k = len(self.anchor_boxes)

        self.separate_anchor_boxes = Rearrange("b (h) (w) (c s) -> b h w c s", s = 4) # rearrange the tensor so bounding box coordinates are on a new dimension 

        self.classifiers = nn.ModuleDict()
        self.transformations = {}
        for i, size in enumerate(self.anchor_boxes):
            # each classifier attends to the section corresponding to the original image, as well as an upsampled feature map
            self.transformations[str(i)] = v2.Compose([torchvision.transforms.Resize(size=(patch_height * size[0], patch_width * size[1]), 
                                                                                      antialias=True)])
            self.classifiers[str(i)] = classification_ViT(image_size = (patch_height * size[0], patch_width * size[1]), 
                                                             patch_size= (classifier_patch_height, classifier_patch_width), 
                                                             encoding_dimensionality = encoding_dimensionality, # dimensionality of the final output
                                                             dim = dim, 
                                                             depth = classifier_depth, 
                                                             heads = classifier_heads, 
                                                             mlp_dim=classifier_mlp_dim,
                                                             channels= dim + image_channels)
            
    def preprocess_slices(self, tensor, index_scalar, features_and_image, device, stream):
        tensor = probabilistic_round(tensor.clone().detach())

        tensor_list = []
        b = tensor.shape[0]
        for i in range(b):
            indexing_tensor = tensor[i, :]
            width_start = indexing_tensor[0]
            height_start = indexing_tensor[1]
            width_end = indexing_tensor[2]
            height_end = indexing_tensor[3]
            batch_location = indexing_tensor[4]
            section = features_and_image[batch_location, :, height_start:height_end + self.patch_height, width_start:width_end + self.patch_width]
            reshaped = self.transformations[str(index_scalar)](section)
            tensor_list.append(reshaped)

        stacked_tensor = torch.stack(tensor_list, dim=0)
        print(f"STACKED TENSOR SHAPE {index_scalar}: ", stacked_tensor.shape)
        if device == "cuda":
            with torch.cuda.stream(stream):
                model_outputs = self.classifiers[str(index_scalar)](stacked_tensor)
        else:
            model_outputs = self.classifiers[str(index_scalar)](stacked_tensor)
        
        return model_outputs

    def forward(self, img, classifier_batch_size = 1024, is_object_threshold = 0.5, device = "cpu"):
        #coarse_regressor_epsilon = 0.05
        # Images come in in shape (channel, width, height)
        # coco labels are in (top_left_corner_height_location, top_left_corner_width_location, rectangle_height, rectangle_width)
        b, c, h, w = img.shape
        feature_map = self.feature_extraction_ViT(img)
        regs, is_object = self.RPN(feature_map)
        is_object = is_object.unsqueeze(4)
        
        
        regs = self.separate_anchor_boxes(regs)
        feature_h = regs.shape[1]
        feature_w = regs.shape[2]

        k_index = torch.arange(0, 9, device = device)
        k_index = repeat(k_index, 'k -> b h w k x', b = b, h = feature_h, w = feature_w, x = 1)
        
        # convert relative widths_coordinates to absolute width
        width_coordinates = torch.arange(start = 0, end = w, step = self.patch_width, device = device)
        width_coordinates = repeat(width_coordinates, 'w -> b h w k x', b = b, h = feature_h, k = self.k, x = 1)
        width_offset = regs[:, :, :, :, (0, )]
        width_coordinates = width_coordinates + width_offset

        # convert height_coordinates to absolute height
        height_coordinates = torch.arange(start = 0, end = h, step = self.patch_height, device = device)
        height_coordinates = repeat(height_coordinates, 'h -> b h w k x', b = b, w = feature_w, k = self.k, x = 1)
        height_offset = regs[:, :, :, :, (1, )] # slice with 
        height_coordinates = height_coordinates + height_offset
        

         # compute widths after being rescaled by anchor box widths
        box_width_ratios = torch.tensor(self.anchor_boxes_width)
        box_width_relative = regs[:, :, :, :, (2, )]
        box_width_ratios = rearrange(box_width_ratios, 'c -> 1 1 1 c 1')
        width_scaled = torch.mul(box_width_ratios, box_width_relative)
        width_abs = width_coordinates + width_scaled
        


        # compute the heights after being rescaled by anchor box sizes
        box_height_ratios = torch.tensor(self.anchor_boxes_height)
        box_height_relative = regs[:, :, :, :, (3, )]
        box_height_ratios = rearrange(box_height_ratios, 'c -> 1 1 1 c 1')
        height_abs = torch.mul(box_height_ratios, box_height_relative) + height_coordinates

        batch_sizes = torch.arange(b, device=  device)
        broadcasted_batch = repeat(batch_sizes, "b -> b h w k x", h = feature_h, w = feature_w, k = self.k, x = 1)


        # After the concatenation, we have upper_left_x, upper_left_y, lower_left_x, lower_left_y, batch_index, is_object, classifier_index
        merged_tensor = torch.cat((width_coordinates, height_coordinates, width_abs, height_abs, broadcasted_batch, is_object, k_index), dim= 4)
        tensor_reshaped = rearrange(merged_tensor, "b h w k x -> (b h w k) x")
        print(tensor_reshaped.shape)

        # Make a mask that throws out all tensors that reach out of the screen, and all tensors where 
        # the lower right corner is higher/more left than the upper left corner
        valid_boxes_mask = ((tensor_reshaped[..., 0] >= 0) & (tensor_reshaped[..., 0] < w) & (tensor_reshaped[..., 1] >= 0) &
         (tensor_reshaped[..., 1] < h) & (tensor_reshaped[..., 2] >= tensor_reshaped[..., 0]) & 
         (tensor_reshaped[..., 2] <= w) & (tensor_reshaped[..., 3] >= tensor_reshaped[..., 1]) & 
         (tensor_reshaped[..., 3] <= h))
        filtered_tensor = tensor_reshaped[valid_boxes_mask]

        is_object_mask = filtered_tensor[..., 5] > is_object_threshold # check on the objectness score
        objects = filtered_tensor[is_object_mask]
        backgrounds = filtered_tensor[~is_object_mask]

        
        num_object_samples = min(objects.shape[0], classifier_batch_size // 2)

        object_indices = torch.randperm(objects.shape[0])[:num_object_samples]
        object_batch = objects[object_indices]

        num_background_samples = classifier_batch_size - num_object_samples
        background_indices = torch.randperm(backgrounds.shape[0])[:num_background_samples]
        background_batch = backgrounds[background_indices]

        classifier_batch = torch.concat((object_batch, background_batch), dim = 0)
        # sort the batch by it's corresponding k value so we can batch feed them into the same classifier
        sorted_indices = torch.argsort(classifier_batch[:, 6])
        classifier_batch = classifier_batch[sorted_indices]
        print(classifier_batch)
        print(classifier_batch.shape)

        # seperate the tensors into lists based on k values
        classifier_batch_list = []
        running_k = round(classifier_batch[0, 6].item())
        start_index = 0
        length = classifier_batch.shape[0]
        for i in range(length):
            current_value = round(classifier_batch[i, 6].item())
            if (current_value != running_k):
                print(current_value, running_k)
                classifier_batch_list.append([classifier_batch[start_index: i, :], running_k])
                start_index = i
                running_k = current_value
        classifier_batch_list.append([classifier_batch[start_index:, :], running_k])

        upsampled_map = torch.nn.functional.interpolate(feature_map, (h, w))
        features_and_image = torch.concat((upsampled_map, img), dim=1)
        # Pad the image so the per pixel offsets work (there's probably a better solution for this)
        # But I don't want to do more math today, and it's probably not worth optimizing
        features_and_image = torch.nn.functional.pad(features_and_image, (8, 8, 8, 8), "constant", 0)

        model_outputs_class = []
        model_outputs_regression = []

        streams = []
        if device == "cuda":
            for i in range(len(classifier_batch_list)):
                streams.append(torch.cuda.Stream())
        else:
            for i in range(len(classifier_batch_list)):
                streams.append(None)
        for i in range(len(classifier_batch_list)):
            outputs = self.preprocess_slices(classifier_batch_list[i][0], index_scalar = classifier_batch_list[i][1], features_and_image=features_and_image, device = device, stream = streams[i])
            model_outputs_class.append(outputs[0])
            model_outputs_regression.append(outputs[1])
            
        if device == "cuda":
            torch.cuda.synchronize()

        class_outputs = torch.cat(model_outputs_class, dim = 0)
        regression_outputs = torch.cat(model_outputs_regression, dim = 0)
        print(class_outputs.shape, regression_outputs.shape)
        exit()
        
        
        print(torch.sum(mask))
        print(locations)
        print(locations.shape)
        exit()
        print(regs.shape, is_object.shape)
        is_object_dict = []
        
        print(features_and_image.shape)

        for i, size in enumerate(self.anchor_boxes):
            objectness = is_object[:, :, :, i:(i + 2)]
            print(objectness.shape)
        exit()
        anchor_boxes = self.RPN.anchor_boxes
