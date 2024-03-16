import torch
import torch.nn.parallel
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F
from multiprocessing import Process

def process_tile(k, i, j, tile_size, pc, symm, xy_stack, indices, PSF, rendered_image):
    image_tile = torch.zeros((tile_size, tile_size), device="cuda")
    product = torch.tensor(1, device="cuda")
    summation = torch.tensor(0, device="cuda")
    center_x = (i * tile_size + (i * tile_size + tile_size - 1)) / 2
    center_y = (j * tile_size + (j * tile_size + tile_size - 1)) / 2
    mask = torch.sqrt((pc.get_xyz[:, 0] - center_x) ** 2 + (pc.get_xyz[:, 1] - center_y) ** 2) >= 25
    for gid in range(pc.get_xyz.shape[0]):
        index = indices[gid]
        if mask[index]:
            # print('continue')
            continue
        xyz = pc.get_xyz[index, :]
        covariance_2D = torch.zeros((2, 2), device="cuda")
        covariance_2D[0, 0] = symm[index, 0]
        covariance_2D[0, 1] = symm[index, 1]
        covariance_2D[1, 0] = symm[index, 1]
        covariance_2D[1, 1] = symm[index, 3]
        opacity = pc.get_opacity[index]
        grey = pc.get_grey[index]
        dist = xy_stack[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size, :] - xyz[:2]
        before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
            (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
                                     dist.unsqueeze(-1))).squeeze()
        after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
                              PSF[k, :, :].unsqueeze(0).unsqueeze(0),
                              padding=17)
        image_tile = image_tile + product * opacity * after_conv.squeeze()
        summation = summation + opacity
        product = product * (1 - opacity)
        if summation >= 1:
            rendered_image[k, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = image_tile[:, :]
            break