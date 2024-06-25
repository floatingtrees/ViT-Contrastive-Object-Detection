from RPN import RPN
import torch


feature_map_shape = (64, 1024, 1024)
model = RPN(feature_map_shape, projection_dim=512)

feature_map = torch.randn(*feature_map_shape).unsqueeze(0)
x, y = model(feature_map)
print(x.shape, y.shape)