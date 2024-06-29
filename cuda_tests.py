import torch

print(torch.cuda.is_available())

x = torch.zeros((100, ))
x.to("cuda")