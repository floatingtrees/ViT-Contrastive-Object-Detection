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

device = "cuda"
model = Detector(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, 
            heads = heads, mlp_dim = mlp_dim)

k = 0
for thing in model.parameters():
    k += thing.numel()
print(k)




preproc = v2.Compose([
    v2.PILToTensor(),
    torchvision.transforms.Resize(size=(1024, 1024), antialias=True),
    ])
#x = Image.open("../thing.jpg")
#img = preproc(x).unsqueeze(0).float()
img = torch.randn(2, 3, 1024, 1024)
img = img.to(device)
y = model(img)
exit()
print(y[0].shape, y[1].shape)



def convertToImage(tensor):
    image = tensor[0, :, :, :].detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    return Image.fromarray((image * 255).astype(np.uint8))
exit()
im2 = convertToImage(z)
plt.imshow(im2)
plt.show()
exit()


