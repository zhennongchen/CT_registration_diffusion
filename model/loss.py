import torch
import torch.nn as nn
import torch.nn.functional as F


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

        loss = (dx.mean() + dy.mean() + dz.mean()) / 3
        return loss
    

class NCCLoss(nn.Module):
    def __init__(self, win=9, eps=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps

    def forward(self, I, J):
        """
        I, J: [B, 1, X, Y, Z]
        """
        ndims = 3
        sum_filt = torch.ones([1, 1] + [self.win] * ndims, device=I.device)

        pad = self.win // 2
        stride = 1

        # local sums
        I_sum = F.conv3d(I, sum_filt, stride=stride, padding=pad)
        J_sum = F.conv3d(J, sum_filt, stride=stride, padding=pad)

        I2_sum = F.conv3d(I * I, sum_filt, stride=stride, padding=pad)
        J2_sum = F.conv3d(J * J, sum_filt, stride=stride, padding=pad)

        IJ_sum = F.conv3d(I * J, sum_filt, stride=stride, padding=pad)

        win_size = self.win ** ndims

        # means
        I_mean = I_sum / win_size
        J_mean = J_sum / win_size

        # cross term
        cross = IJ_sum - J_mean * I_sum - I_mean * J_sum + I_mean * J_mean * win_size

        # variances
        I_var = I2_sum - 2 * I_mean * I_sum + I_mean * I_mean * win_size
        J_var = J2_sum - 2 * J_mean * J_sum + J_mean * J_mean * win_size

        # NCC
        ncc = cross * cross / (I_var * J_var + self.eps)

        # LOSS = negative NCC
        return -torch.mean(ncc)