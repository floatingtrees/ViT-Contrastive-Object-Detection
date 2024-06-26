import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from torchvision.transforms import v2
from einops.layers.torch import Rearrange
import numpy as np



preproc = v2.Compose([
    v2.PILToTensor(),
    torchvision.transforms.Resize(size=(256, 256), antialias=True),
    ])
x = Image.open("../thing.jpg")
img = preproc(x).unsqueeze(0)
print(img.shape)
layer = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 1, p2 = 1, h = 256)
y = layer(img)
print(y.shape)
l2 = Rearrange('b (h w) (f) -> b (f) (h) (w)',h = 256)
z = l2(y)
print(z.shape)
def convertToImage(tensor):
    image = tensor[0, :, :, :].detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return Image.fromarray((image * 255).astype(np.uint8))

im2 = convertToImage(z)
plt.imshow(im2)
plt.show()
exit()


