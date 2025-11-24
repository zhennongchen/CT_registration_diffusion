import torch
import torch.nn as nn

class GradSmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, MVF):
        """
        MVF: (b, 3, x, y, z)
        """
        dx = torch.abs(MVF[:, :, 1:, :, :] - MVF[:, :, :-1, :, :])
        dy = torch.abs(MVF[:, :, :, 1:, :] - MVF[:, :, :, :-1, :])
        dz = torch.abs(MVF[:, :, :, :, 1:] - MVF[:, :, :, :, :-1])

        loss = (dx.mean() + dy.mean() + dz.mean())
        return loss
