import torch
from torch import nn
from einops.layers.torch import Rearrange


class RPN(nn.Module):
    def __init__(self, feature_map_size, projection_dim,  box_heights = (1, 2, 4), box_widths = (1, 2, 4)):
        super(RPN, self).__init__()
        c, h, w = feature_map_size
        self.anchor_boxes = []
        for i in box_heights:
            for j in box_widths:
                self.anchor_boxes.append((i, j))
        self.k = len(self.anchor_boxes)
        self.kernel = nn.Conv2d(c, projection_dim, 3, padding = "same")
        self.sigmoid = nn.Sigmoid()
        self.cls = nn.Linear(projection_dim, self.k) # 0 is object, 1 is not-object
        self.relu = nn.ReLU()
        self.reg = nn.Linear(projection_dim, 4 * self.k)
        self.to_channels_last = Rearrange('b c (h) (w) -> b h w c')

    def forward(self, feature_map):
        x = self.kernel(feature_map)
        x = self.relu(x)
        x = self.to_channels_last(x)

        print(x.shape)
        regs = self.reg(x)
        is_object = self.sigmoid(self.cls(x))

        return regs, is_object
    
    def get_anchor_boxes(self):
        return self.anchor_boxes
