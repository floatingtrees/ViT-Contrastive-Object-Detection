import torch
import numpy as np
from PIL import Image, ImageDraw
from random import randrange

def convertToImage(tensor):
    image = tensor[0, :, :, :].cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return Image.fromarray((image * 255).astype(np.uint8))


def visualize_image(img_tensor, boxes, category_encodings, objectness, iteration):
     color_list = ["pink", "red", "teal", "blue", "orange", "yellow", "black", "magenta","green","aqua"]*10
     img = convertToImage(img_tensor)
     draw = ImageDraw.Draw(img)
     for i in range(boxes.shape[0]):
         if objectness[i].item() > 0.7:
             x = boxes[i, 0]
             y = boxes[i, 1]
             w = boxes[i, 2]
             h = boxes[i, 3]
             # coordinates outputted by the model are absolute
             box_coords = (x, y, w, h)
             draw.rectangle(box_coords, outline=color_list[randrange(len(color_list))], width=2)
     img.save(f"../visualizations/img_{iteration}.jpg")

    



def calculate_reverse_overlap(A):
    B_first_column = torch.clamp(A[:, 0] - A[:, 2], 0) # Upper_left_x - lower_right_x should always be negative
    # Adds is penalty for edges crossing over
    B_second_column = torch.clamp(A[:, 1] - A[:, 3], 0)
    invalid_total = B_first_column + B_second_column
    return invalid_total

def calculate_iou(A, B):
    A_exp = A[:, None, :]  # Shape: (num_boxes, 1, 4)
    B_exp = B[None, :, :]  # Shape: (1, batch_size, 4)

    # Compute the coordinates of the intersection boxes
    max_ul = torch.max(A_exp[:, :, :2], B_exp[:, :, :2])  # upper-left corner
    min_lr = torch.min(A_exp[:, :, 2:], B_exp[:, :, 2:])  # lower-right corner

    # Compute the sizes of the intersection boxes
    inter_sizes = torch.clamp(min_lr - max_ul, min=0)  # (num_boxes, batch_size, 2)
    inter_area = inter_sizes[:, :, 0] * inter_sizes[:, :, 1]  # (num_boxes, batch_size)

    # Compute the area of each box in A and B
    A_area = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])  # (num_boxes,)
    B_area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])  # (batch_size,)

    # Expand areas for broadcasting
    A_area_exp = A_area[:, None]  # (num_boxes, 1)
    B_area_exp = B_area[None, :]  # (1, batch_size)

    # Compute the union area
    union_area = A_area_exp + B_area_exp - inter_area

    # Compute the IoU
    iou = inter_area / union_area
    return iou
