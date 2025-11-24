
import torch
from torch import nn, einsum
import torch.nn.functional as F

def warp_from_mvf(seg_t0, mvf_voxel):
    """
    seg_t0:    [B, 1, X, Y, Z] → will be permuted to [B, 1, Z, Y,X]
    mvf_voxel: [B, 3, X, Y, Z] → will be permuted to [B, 3, Z, Y,X]
    returns:
        warped_seg: [B, 1, X, Y, Z]
    """
    # Step 0: permute to [B, 1, Z, Y,X]
    seg_t0 = seg_t0.permute(0, 1, 4, 3, 2).contiguous()
    mvf_voxel = mvf_voxel.permute(0, 1, 4, 3, 2).contiguous()

    B, _, D, H, W = seg_t0.shape  # Z, Y, X
    device = seg_t0.device

    # Step 1: normalize MVF to [-1, 1]
    mvf_norm = torch.zeros_like(mvf_voxel)
    mvf_norm[:, 0] = mvf_voxel[:, 0] * 2 / (W - 1)  # dx
    mvf_norm[:, 1] = mvf_voxel[:, 1] * 2 / (H - 1)  # dy
    mvf_norm[:, 2] = mvf_voxel[:, 2] * 2 / (D - 1)  # dz

    # Step 2: create identity grid in Z, Y, X order
    grid_z = torch.linspace(-1, 1, D, device=device)
    grid_y = torch.linspace(-1, 1, H, device=device)
    grid_x = torch.linspace(-1, 1, W, device=device)
    meshz, meshy, meshx = torch.meshgrid(grid_z, grid_y, grid_x, indexing='ij')  # [D, H, W]
    identity_grid = torch.stack((meshx, meshy, meshz), dim=-1)  # [D, H, W, 3]
    identity_grid = identity_grid.unsqueeze(0).repeat(B, 1, 1, 1, 1)  # [B, D, H, W, 3]

    # Step 3: add displacement
    displacement_grid = identity_grid + mvf_norm.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

    # Step 4: warp
    warped = F.grid_sample(
        seg_t0, displacement_grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )  # [B, 1, D, H, W]

    # Step 5: permute back to [B, 1, X, Y, Z]
    warped = warped.permute(0, 1, 4,3, 2).contiguous()

    return warped  # [B, 1, X, Y, Z]