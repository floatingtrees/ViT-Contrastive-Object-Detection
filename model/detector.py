import torch
from torch import nn
from RPN import RPN
from ViT import feature_extraction_ViT, classification_ViT
from einops.layers.torch import Rearrange
from einops import repeat

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
                classifier_patch_size = 1, # classifier patch size, with respect to the feature map
                box_heights = (2, 4, 8), #possible anchor box heights for RPN
                box_widths = (2, 4, 8) # possible anchor box widths for RPN
                ):
        super(Detector, self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.image_dim = (image_channels, image_height, image_width)
        self.patch_height = patch_height
        self.patch_width = patch_width

        classifier_patch_height, classifier_patch_width = pair(classifier_patch_size)
        feature_map_shape = (dim, image_height// patch_height, image_width // patch_width)
        
        self.feature_extraction_ViT = feature_extraction_ViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
            heads = heads, mlp_dim = mlp_dim, channels=image_channels)
        
        self.seperate_anchor_boxes = Rearrange("b (h) (w) (c s) -> b h w c s", s = 4) # rearrange the tensor so bounding box coordinates are seperated out 

        self.RPN = RPN(feature_map_shape, projection_dim=dim)
        self.classifiers = nn.ModuleDict()
        self.anchor_boxes = self.RPN.anchor_boxes
        self.k = len(self.anchor_boxes)
        for size in self.anchor_boxes:
            # each classifier attends to the section corresponding to the original image, as well as an upsampled feature map
            self.classifiers[str(size)] = classification_ViT(image_size = (patch_height * size[0], patch_width * size[1]), 
                                                             patch_size= (classifier_patch_height, classifier_patch_width), 
                                                             encoding_dimensionality = encoding_dimensionality, # dimensionality of the final output
                                                             dim = dim, 
                                                             depth = classifier_depth, 
                                                             heads = classifier_heads, 
                                                             mlp_dim=classifier_mlp_dim,
                                                             channels= dim + image_channels)

    def forward(self, img, classifier_batch_size = 64):
        b, c, h, w = img.shape
        feature_map = self.feature_extraction_ViT(img)
        regs, is_object = self.RPN(feature_map)
        
        regs = self.seperate_anchor_boxes(regs)
        feature_h = regs.shape[1]
        feature_w = regs.shape[2]
        # convert height_coordinates to absolute height

        height_coordinates = torch.arange(start = 0, end = h, step = self.patch_height)
        height_coordinates = repeat(height_coordinates, 'h -> b h w k x', b = b, w = feature_w, k = self.k, x = 1)
        height_offset = regs[:, :, :, :, (0, )] # slice with 
        height_coordinates = height_coordinates + height_offset

        # convert relative widths_coordinates to absolute widths
        width_coordinates = torch.arange(start = 0, end = w, step = self.patch_width)
        width_coordinates = repeat(width_coordinates, 'w -> b h w k x', b = b, h = feature_h, k = self.k, x = 1)
        width_offset = regs[:, :, :, :, (1, )]
        width_coordinates = width_coordinates + width_offset

        

        box_height = regs[:, :, :, :, 2]
        box_width = regs[:, :, :, :, 3]
        exit()
        mask = (is_object > 0.5)
        locations = torch.nonzero(mask)
        print(torch.sum(mask))
        print(locations)
        print(locations.shape)
        exit()
        print(regs.shape, is_object.shape)
        is_object_dict = []
        print(feature_map.shape)
        upsampled_map = torch.nn.functional.interpolate(feature_map, (h, w))
        features_and_image = torch.concat((upsampled_map, img), dim=1)
        print(features_and_image.shape)

        for i, size in enumerate(self.anchor_boxes):
            objectness = is_object[:, :, :, i:(i + 2)]
            print(objectness.shape)
        exit()
        anchor_boxes = self.RPN.anchor_boxes
