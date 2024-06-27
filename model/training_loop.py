import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
import numpy as np
from detector import Detector

image_size = 1024
patch_size = 16
num_classes = 100
dim = 512
depth = 1
heads = 2
mlp_dim = 2048

model = Detector(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
            heads = heads, mlp_dim = mlp_dim)

k = 0
for thing in model.parameters():
    k += thing.numel()
print(k)

