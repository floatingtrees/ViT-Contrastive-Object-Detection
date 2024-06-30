import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from detector import Detector
from utils import calculate_iou, calculate_reverse_overlap, visualize_image
from model_dataloader import BoxDataset
from torch.utils.data import DataLoader
import time
from torch.cuda.amp import GradScaler 
import gc

if __name__ == "__main__":
    image_size = 1024
    patch_size = 16
    dim = 512
    depth = 4
    heads = 16
    mlp_dim = 16384
    print_period = 100
    model = Detector(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
                heads = heads, mlp_dim = mlp_dim)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    coco_annotations_file="../../coco2017/annotations/instances_train2017.json"
    coco_images_dir="../../coco2017/train2017"
    dataset = BoxDataset(coco_annotations_file, coco_images_dir, device = "cuda")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers = 2)
    model.to(device)
    k = 0
    for thing in model.parameters():
        k += thing.numel()
    print(k)

    optimizer = torch.optim.Adam(model.parameters())

    binary_crossentropy_loss_fn = torch.nn.BCELoss()

    num_epochs = 10
    scaler = GradScaler()


    for epoch in range(num_epochs):
        running_losses = [0, 0, 0, 0]
        for i, batch  in enumerate(dataloader):
            optimizer.zero_grad()
            img, boxes, categories, original = batch 
            if img.shape[1] != 3:
                print("Degenerate tensor detected\n\n")
                continue
            img = img.to(device)
            boxes = boxes.to(device)
            categories = categories.to(device)
            if boxes.numel() == 0: # Better way to handle empty images?
                continue
            class_outputs, adjusted_regression_outputs, objectness = model(img, classifier_batch_size = 1024, device = device)
            
            invalid_total = calculate_reverse_overlap(adjusted_regression_outputs)
            # invalid boxes are really bad, so the model should learn to fix those first
            invalid_penalty = torch.mean(invalid_total) * 50 

            boxes = boxes.squeeze(0)
            categories = categories.squeeze(0)

            IOU_matrix = calculate_iou(adjusted_regression_outputs, boxes)
            index = torch.argmax(IOU_matrix, dim = 0)

            IOU_mask = (IOU_matrix > 0.7)
            index_filler = torch.arange(IOU_matrix.shape[1], device = device)
            IOU_mask[index, index_filler] = 1
            object_label_mask = IOU_mask.clone().float()

            object_labels = torch.clamp(torch.sum(object_label_mask, axis = 1, keepdim = False), 0, 1)
            # Compute objectness loss
            objectness_loss = binary_crossentropy_loss_fn(objectness, object_labels)

            index = torch.argmax(IOU_matrix, dim = 1)
            largest_intersections = torch.zeros_like(IOU_mask, dtype = torch.bool)
            largest_intersections[torch.arange(index.shape[0], device = device), index] = 1
            true_mask = IOU_mask & largest_intersections
            relevant_IOUs = true_mask * IOU_matrix
            # Not sure about the best way to deal with this loss, so just subtract so the optimizer maximizes it
            IOU_boost = -torch.mean(relevant_IOUs)


            class_normed = class_outputs / torch.norm(class_outputs, p=2, dim=1, keepdim=True)
            category_label_normed = categories / torch.norm(categories, p=2, dim=1, keepdim=True)
            similarities = torch.matmul(class_normed, category_label_normed.T)
            masked_similarities = torch.mul(largest_intersections, similarities)
            reduced_similarities = torch.sum(masked_similarities, axis = 1)
            reduced_similarities = reduced_similarities / 2 + 0.5

            class_loss = binary_crossentropy_loss_fn(reduced_similarities, torch.ones_like(reduced_similarities))

            loss = class_loss + IOU_boost + objectness_loss + invalid_penalty

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_losses[0] += class_loss.item()
            running_losses[1] += IOU_boost.item()
            running_losses[2] += objectness_loss.item()
            running_losses[3] += invalid_penalty.item()
            if i % print_period == 0:
                print(f"Epoch {i}")
                print(f"Class loss: {running_losses[0]/print_period}")
                print(f"IOU_boost: {running_losses[1]/print_period}")
                print(f"objectness_loss: {running_losses[2]/print_period}")
                print(f"invalid_penalty: {running_losses[3]/print_period}")
                print("\n\n")
                running_losses = [0, 0, 0, 0]
                visualize_image(original, adjusted_regression_outputs, categories, objectness, i)