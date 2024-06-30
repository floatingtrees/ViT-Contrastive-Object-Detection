import torch

# Define the example tensors A and B
# Tensor A: (num_boxes, 4)
A = torch.tensor([
    [1, 1, 4, 4],
    [5, 5, 8, 8]
])

# Tensor B: (batch_size, 4)
B = torch.tensor([
    [1, 1, 2, 2],
    [2, 2, 6, 6],
    [0, 0, 3, 3]
])

# Expand A and B for broadcasting
A_exp = A[:, None, :]  # Shape: (num_boxes, 1, 4)
B_exp = B[None, :, :]  # Shape: (1, batch_size, 4)
print(A_exp, B_exp)

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

print("IoU Matrix:\n", iou)
