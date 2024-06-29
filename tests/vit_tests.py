import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
import numpy as np
from ViT import ViT

image_size = 1024
patch_size = 16
num_classes = 100
dim = 1024
depth = 8
heads = 16
mlp_dim = 2048

model = ViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
            heads = heads, mlp_dim = mlp_dim)

k = 0
for thing in model.parameters():
    k += thing.numel()
print(k)




preproc = v2.Compose([
    v2.PILToTensor(),
    torchvision.transforms.Resize(size=(1024, 1024), antialias=True),
    ])
x = Image.open("../thing.jpg")
img = preproc(x).unsqueeze(0).float()
print(img.shape)
x = torch.randn(2, 3, 1024, 1024)
y = model(img)
print(y.shape)

def convertToImage(tensor):
    image = tensor[0, :, :, :].detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return Image.fromarray((image * 255).astype(np.uint8))
exit()
im2 = convertToImage(z)
plt.imshow(im2)
plt.show()
exit()


