#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import pdb

import torch
import torch.nn.parallel
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import torch.nn.functional as F
from multiprocessing import Process
import worker_module
import multiprocessing
import threading

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


# def render_psf(pc: GaussianModel, PSF):
#     """
#     Render the scene.
#
#     Background tensor (bg_color) must be on GPU!
#     """
#
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     # try:
#     #     screenspace_points.retain_grad()
#     # except:
#     #     pass
#     # screenspace_points.retain_grad()
#     num_psf = PSF.shape[0]
#     # pdb.set_trace()
#     # screenspace_points[:, 0] = pc.get_xyz[:, 0]
#     # screenspace_points[:, 1] = pc.get_xyz[:, 1]
#     # pdb.set_trace()
#     rendered_image = torch.zeros((num_psf, 512, 512), device="cuda")
#     sorted_tensor, indices = torch.sort(pc.get_xyz[:, 2])
#     symm = pc.get_covariance()
#     x_range = torch.arange(512, device="cuda")
#     y_range = torch.arange(512, device="cuda")
#     y, x = torch.meshgrid(y_range, x_range)
#     for j in range(num_psf):
#         # print("psf_id:")
#         # print(j)
#         image = torch.zeros((512, 512), device="cuda")
#         summation = torch.tensor(0, device="cuda")
#         for i in range(pc.get_xyz.shape[0]):
#             # print("gaussian id:")
#             # print(i)
#             index = indices[i]
#             xyz = pc.get_xyz[index, :]
#             covariance_2D = torch.zeros((2, 2), device="cuda")
#             covariance_2D[0, 0] = symm[index, 0]
#             covariance_2D[0, 1] = symm[index, 1]
#             covariance_2D[1, 0] = symm[index, 1]
#             covariance_2D[1, 1] = symm[index, 3]
#             opacity = pc.get_opacity[index]
#             grey = pc.get_grey[index]
#             # x_range = torch.arange(512, device="cuda")
#             # y_range = torch.arange(512, device="cuda")
#             # y, x = torch.meshgrid(y_range, x_range)
#             dist = torch.stack([x, y], dim=-1) - xyz[:2]
#             T = torch.exp(- summation)
#             before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
#                 (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
#                                          dist.unsqueeze(-1))).squeeze()
#             # before_conv = before_conv.squeeze()
#             after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0), PSF[j, :, :].unsqueeze(0).unsqueeze(0),
#                                   padding=17)
#             image = image + T * (1 - torch.exp(- 0.1 * opacity)) * after_conv.squeeze()
#             summation = summation + opacity * 0.1
#             # torch.cuda.empty_cache()
#         rendered_image[j, :, :] = image[:, :]
#
#     return rendered_image

# def render_psf(pc: GaussianModel, PSF):
#     """
#     Render the scene.
#
#     Background tensor (bg_color) must be on GPU!
#     """
#
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     # try:
#     #     screenspace_points.retain_grad()
#     # except:
#     #     pass
#     # screenspace_points.retain_grad()
#     num_psf = PSF.shape[0]
#     print("num_psf")
#     print(num_psf)
#     num_tiles = 16
#     tile_size = 32
#     # pdb.set_trace()
#     # screenspace_points[:, 0] = pc.get_xyz[:, 0]
#     # screenspace_points[:, 1] = pc.get_xyz[:, 1]
#     # pdb.set_trace()
#     rendered_image = torch.zeros((num_psf, 512, 512), device="cuda")
#     sorted_tensor, indices = torch.sort(pc.get_xyz[:, 2])
#     symm = pc.get_covariance()
#     x_range = torch.arange(512, device="cuda")
#     y_range = torch.arange(512, device="cuda")
#     y, x = torch.meshgrid(y_range, x_range)
#     xy_stack = torch.stack([x, y], dim=-1)
#     flag = torch.zeros((num_tiles, num_tiles), device="cuda")
#     cnt = 0
#     cnt_mask = 0
#     for k in range(num_psf):
#         for i in range(num_tiles):
#             for j in range(num_tiles):
#                 image_tile = torch.zeros((tile_size, tile_size), device="cuda")
#                 product = torch.tensor(1, device="cuda")
#                 summation = torch.tensor(0, device="cuda")
#                 center_x = (i * tile_size + (i * tile_size + tile_size - 1)) / 2
#                 center_y = (j * tile_size + (j * tile_size + tile_size - 1)) / 2
#                 mask = torch.sqrt((pc.get_xyz[:, 0] - center_x) ** 2 + (pc.get_xyz[:, 1] - center_y) ** 2) >= 25
#                 # print(mask)
#                 for gid in range(pc.get_xyz.shape[0]):
#                     index = indices[gid]
#                     if mask[index]:
#                         # print('continue')
#                         continue
#                     flag[i, j] = 1
#                     xyz = pc.get_xyz[index, :]
#                     covariance_2D = torch.zeros((2, 2), device="cuda")
#                     covariance_2D[0, 0] = symm[index, 0]
#                     covariance_2D[0, 1] = symm[index, 1]
#                     covariance_2D[1, 0] = symm[index, 1]
#                     covariance_2D[1, 1] = symm[index, 3]
#                     # print(covariance_2D)
#                     # print(pc.get_xyz)
#                     # print(pc.get_rotation)
#                     # print(pc.get_scaling)
#                     opacity = pc.get_opacity[index]
#                     # print(opacity)
#                     # print(index)
#                     grey = pc.get_grey[index]
#                     dist = xy_stack[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :] - xyz[:2]
#                     before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
#                         (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
#                                                  dist.unsqueeze(-1))).squeeze()
#                     after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
#                                           PSF[k, :, :].unsqueeze(0).unsqueeze(0),
#                                           padding=17)
#                     # T = torch.exp(- summation)
#                     image_tile = image_tile + product * opacity * after_conv.squeeze()
#                     # image_tile = image_tile + T * (1 - torch.exp(- 0.1 * opacity)) * after_conv.squeeze()
#                     summation = summation + opacity
#                     # summation = summation + opacity * 0.1
#                     product = product * (1 - opacity)
#                     cnt_mask += 1
#                     if summation >= 1:
#                         rendered_image[k, i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = image_tile[:, :]
#                         cnt += 1
#                         break
#                 # rendered_image[k, i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = image_tile[:, :]
#
#     print("flag")
#     print(flag.sum())
#     print("cnt")
#     print(cnt)
#     print("cnt_mask")
#     print(cnt_mask)
#     return rendered_image

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
            # return image_tile[:, :]
            break

def render_psf(pc: GaussianModel, PSF):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # try:
    #     screenspace_points.retain_grad()
    # except:
    #     pass
    # screenspace_points.retain_grad()
    num_psf = PSF.shape[0]
    processes = []
    threads = []
    # print("num_psf")
    # print(num_psf)
    num_tiles = 16
    tile_size = 32
    # pdb.set_trace()
    # screenspace_points[:, 0] = pc.get_xyz[:, 0]
    # screenspace_points[:, 1] = pc.get_xyz[:, 1]
    # pdb.set_trace()
    rendered_image = torch.zeros((num_psf, 512, 512), device="cuda")
    sorted_tensor, indices = torch.sort(pc.get_xyz[:, 2])
    symm = pc.get_covariance()
    x_range = torch.arange(512, device="cuda")
    y_range = torch.arange(512, device="cuda")
    y, x = torch.meshgrid(y_range, x_range)
    xy_stack = torch.stack([x, y], dim=-1)
    # flag = torch.zeros((num_tiles, num_tiles), device="cuda")
    # cnt = 0
    # cnt_mask = 0
    # product = torch.ones((num_tiles, num_tiles), device="cuda")
    # summation = torch.zeros((num_tiles, num_tiles), device="cuda")
    i_range = torch.arange(num_tiles, device="cuda")
    j_range = torch.arange(num_tiles, device="cuda")
    j, i = torch.meshgrid(j_range, i_range)
    center_x = (i * tile_size + (i * tile_size + tile_size - 1)) / 2
    center_y = (j * tile_size + (j * tile_size + tile_size - 1)) / 2
    # print("center_x")
    # print(center_x.shape)
    # print("dddd:")
    # print(pc.get_xyz[:, 0].expand(num_tiles, num_tiles, pc.get_xyz.shape[0]).shape)
    # aaa = pc.get_xyz[:, 0].expand(num_tiles, num_tiles, pc.get_xyz.shape[0]) - center_x.unsqueeze(2)
    # mask = (torch.sqrt((pc.get_xyz[:, 0].expand(num_tiles, num_tiles, pc.get_xyz.shape[0]) - center_x.unsqueeze(2)) ** 2 + (pc.get_xyz[:, 1].expand(num_tiles, num_tiles, pc.get_xyz.shape[0]) - center_y.unsqueeze(2)) ** 2) <= 25)

    # 2X2 tiles per iter
    # for k in range(num_psf):
    #     for x_tile in range(8):
    #         for y_tile in range(8):
    #             image_sub = torch.zeros((64, 64), device="cuda")
    #             product = torch.ones((2, 2), device="cuda")
    #             summation = torch.zeros((2, 2), device="cuda")
    #             mask = (torch.sqrt((pc.get_xyz[:, 0].expand(2, 2, pc.get_xyz.shape[0]) - center_x[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(2)) ** 2 + (pc.get_xyz[:, 1].expand(2, 2, pc.get_xyz.shape[0]) - center_y[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(2)) ** 2) <= 25)
    #             for gid in range(pc.get_xyz.shape[0]):
    #                 index = indices[gid]
    #                 if mask[:, :, index].float().sum() == 0.0:
    #                     continue
    #                 xyz = pc.get_xyz[index, :]
    #                 covariance_2D = torch.zeros((2, 2), device="cuda")
    #                 covariance_2D[0, 0] = symm[index, 0]
    #                 covariance_2D[0, 1] = symm[index, 1]
    #                 covariance_2D[1, 0] = symm[index, 1]
    #                 covariance_2D[1, 1] = symm[index, 3]
    #                 opacity = pc.get_opacity[index]
    #                 grey = pc.get_grey[index]
    #                 dist = xy_stack[x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] - xyz[:2]
    #                 before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
    #                     (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
    #                                              dist.unsqueeze(-1))).squeeze()
    #                 after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
    #                                       PSF[k, :, :].unsqueeze(0).unsqueeze(0),
    #                                       padding=17)
    #                 # rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] = rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] + F.interpolate(
    #                 #     (summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] < 1).float().unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0),
    #                 #                                               size=(64, 64),
    #                 #                                               mode='nearest').squeeze() * F.interpolate(
    #                 #     product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 image_sub = image_sub + F.interpolate(
    #                     (summation < 1).float().unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest').squeeze() * F.interpolate(
    #                     product.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 # summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] + opacity * mask[:, :, index]
    #                 summation = summation + opacity * mask[:, :, index]
    #                 # product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 product = product * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 if (summation < 1).float().sum() == 0.0:
    #                     rendered_image[k, x_tile * 64:(x_tile * 64 + 64), y_tile * 64:(y_tile * 64 + 64)] = image_sub[:, :]
    #                     break
    #                 if gid == (pc.get_xyz.shape[0] - 1):
    #                     rendered_image[k, x_tile * 64:(x_tile * 64 + 64), y_tile * 64:(y_tile * 64 + 64)] = image_sub[:, :]
    #                     break


    # 4X4 tiles per iter
    # for k in range(num_psf):
    #     for x_tile in range(4):
    #         for y_tile in range(4):
    #             image_sub = torch.zeros((128, 128), device="cuda")
    #             product = torch.ones((4, 4), device="cuda")
    #             summation = torch.zeros((4, 4), device="cuda")
    #             mask = (torch.sqrt((pc.get_xyz[:, 0].expand(4, 4, pc.get_xyz.shape[0]) - center_x[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)].unsqueeze(2)) ** 2 + (pc.get_xyz[:, 1].expand(4, 4, pc.get_xyz.shape[0]) - center_y[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)].unsqueeze(2)) ** 2) <= 25)
    #             for gid in range(pc.get_xyz.shape[0]):
    #                 index = indices[gid]
    #                 if mask[:, :, index].float().sum() == 0.0:
    #                     continue
    #                 xyz = pc.get_xyz[index, :]
    #                 covariance_2D = torch.zeros((2, 2), device="cuda")
    #                 covariance_2D[0, 0] = symm[index, 0]
    #                 covariance_2D[0, 1] = symm[index, 1]
    #                 covariance_2D[1, 0] = symm[index, 1]
    #                 covariance_2D[1, 1] = symm[index, 3]
    #                 opacity = pc.get_opacity[index]
    #                 grey = pc.get_grey[index]
    #                 dist = xy_stack[x_tile*128:(x_tile*128+128), y_tile*128:(y_tile*128+128)] - xyz[:2]
    #                 before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
    #                     (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
    #                                              dist.unsqueeze(-1))).squeeze()
    #                 after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
    #                                       PSF[k, :, :].unsqueeze(0).unsqueeze(0),
    #                                       padding=17)
    #                 # rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] = rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] + F.interpolate(
    #                 #     (summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] < 1).float().unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0),
    #                 #                                               size=(64, 64),
    #                 #                                               mode='nearest').squeeze() * F.interpolate(
    #                 #     product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 image_sub = image_sub + F.interpolate(
    #                     (summation < 1).float().unsqueeze(0).unsqueeze(0), size=(128, 128),
    #                     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0), size=(128, 128), mode='nearest').squeeze() * F.interpolate(
    #                     product.unsqueeze(0).unsqueeze(0), size=(128, 128), mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 # summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] + opacity * mask[:, :, index]
    #                 summation = summation + opacity * mask[:, :, index]
    #                 # product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 product = product * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 if (summation < 1).float().sum() == 0.0:
    #                     rendered_image[k, x_tile * 128:(x_tile * 128 + 128), y_tile * 128:(y_tile * 128 + 128)] = image_sub[:, :]
    #                     break
    #                 if gid == (pc.get_xyz.shape[0] - 1):
    #                     rendered_image[k, x_tile * 128:(x_tile * 128 + 128), y_tile * 128:(y_tile * 128 + 128)] = image_sub[:, :]
    #                     break


    # 8X8 tiles per iter
    for k in range(num_psf):
        for x_tile in range(2):
            for y_tile in range(2):
                image_sub = torch.zeros((256, 256), device="cuda")
                product = torch.ones((8, 8), device="cuda")
                summation = torch.zeros((8, 8), device="cuda")
                mask = (torch.sqrt((pc.get_xyz[:, 0].expand(8, 8, pc.get_xyz.shape[0]) - center_x[x_tile*8:(x_tile*8+8), y_tile*8:(y_tile*8+8)].unsqueeze(2)) ** 2 + (pc.get_xyz[:, 1].expand(8, 8, pc.get_xyz.shape[0]) - center_y[x_tile*8:(x_tile*8+8), y_tile*8:(y_tile*8+8)].unsqueeze(2)) ** 2) <= 25)
                for gid in range(pc.get_xyz.shape[0]):
                    index = indices[gid]
                    if mask[:, :, index].float().sum() == 0.0:
                        continue
                    xyz = pc.get_xyz[index, :]
                    covariance_2D = torch.zeros((2, 2), device="cuda")
                    covariance_2D[0, 0] = symm[index, 0]
                    covariance_2D[0, 1] = symm[index, 1]
                    covariance_2D[1, 0] = symm[index, 1]
                    covariance_2D[1, 1] = symm[index, 3]
                    opacity = pc.get_opacity[index]
                    grey = pc.get_grey[index]
                    dist = xy_stack[x_tile*256:(x_tile*256+256), y_tile*256:(y_tile*256+256)] - xyz[:2]
                    before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
                        (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
                                                 dist.unsqueeze(-1))).squeeze()
                    after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
                                          PSF[k, :, :].unsqueeze(0).unsqueeze(0),
                                          padding=17)
                    # rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] = rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] + F.interpolate(
                    #     (summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] < 1).float().unsqueeze(0).unsqueeze(0), size=(64, 64),
                    #     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0),
                    #                                               size=(64, 64),
                    #                                               mode='nearest').squeeze() * F.interpolate(
                    #     product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(0).unsqueeze(0), size=(64, 64),
                    #     mode='nearest').squeeze() * opacity * after_conv.squeeze()
                    image_sub = image_sub + F.interpolate(
                        (summation < 1).float().unsqueeze(0).unsqueeze(0), size=(256, 256),
                        mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest').squeeze() * F.interpolate(
                        product.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest').squeeze() * opacity * after_conv.squeeze()
                    # summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] + opacity * mask[:, :, index]
                    summation = summation + opacity * mask[:, :, index]
                    # product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] * (1 - opacity + opacity * (~ mask[:, :, index]))
                    product = product * (1 - opacity + opacity * (~ mask[:, :, index]))
                    if (summation < 1).float().sum() == 0.0:
                        rendered_image[k, x_tile * 256:(x_tile * 256 + 256), y_tile * 256:(y_tile * 256 + 256)] = image_sub[:, :]
                        break
                    if gid == (pc.get_xyz.shape[0] - 1):
                        rendered_image[k, x_tile * 256:(x_tile * 256 + 256), y_tile * 256:(y_tile * 256 + 256)] = image_sub[:, :]
                        break


    # 16X16 tiles per iter
    # for k in range(num_psf):
    #     for x_tile in range(1):
    #         for y_tile in range(1):
    #             image_sub = torch.zeros((512, 512), device="cuda")
    #             product = torch.ones((16, 16), device="cuda")
    #             summation = torch.zeros((16, 16), device="cuda")
    #             mask = (torch.sqrt((pc.get_xyz[:, 0].expand(16, 16, pc.get_xyz.shape[0]) - center_x[x_tile*16:(x_tile*16+16), y_tile*16:(y_tile*16+16)].unsqueeze(2)) ** 2 + (pc.get_xyz[:, 1].expand(16, 16, pc.get_xyz.shape[0]) - center_y[x_tile*16:(x_tile*16+16), y_tile*16:(y_tile*16+16)].unsqueeze(2)) ** 2) <= 25)
    #             for gid in range(pc.get_xyz.shape[0]):
    #                 index = indices[gid]
    #                 if mask[:, :, index].float().sum() == 0.0:
    #                     continue
    #                 xyz = pc.get_xyz[index, :]
    #                 covariance_2D = torch.zeros((2, 2), device="cuda")
    #                 covariance_2D[0, 0] = symm[index, 0]
    #                 covariance_2D[0, 1] = symm[index, 1]
    #                 covariance_2D[1, 0] = symm[index, 1]
    #                 covariance_2D[1, 1] = symm[index, 3]
    #                 opacity = pc.get_opacity[index]
    #                 grey = pc.get_grey[index]
    #                 dist = xy_stack[x_tile*512:(x_tile*512+512), y_tile*512:(y_tile*512+512)] - xyz[:2]
    #                 before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
    #                     (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
    #                                              dist.unsqueeze(-1))).squeeze()
    #                 after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
    #                                       PSF[k, :, :].unsqueeze(0).unsqueeze(0),
    #                                       padding=17)
    #                 # rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] = rendered_image[k, x_tile*64:(x_tile*64+64), y_tile*64:(y_tile*64+64)] + F.interpolate(
    #                 #     (summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] < 1).float().unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0),
    #                 #                                               size=(64, 64),
    #                 #                                               mode='nearest').squeeze() * F.interpolate(
    #                 #     product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)].unsqueeze(0).unsqueeze(0), size=(64, 64),
    #                 #     mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 image_sub = image_sub + F.interpolate(
    #                     (summation < 1).float().unsqueeze(0).unsqueeze(0), size=(512, 512),
    #                     mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze() * F.interpolate(
    #                     product.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 # summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = summation[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] + opacity * mask[:, :, index]
    #                 summation = summation + opacity * mask[:, :, index]
    #                 # product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] = product[x_tile*2:(x_tile*2+2), y_tile*2:(y_tile*2+2)] * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 product = product * (1 - opacity + opacity * (~ mask[:, :, index]))
    #                 if (summation < 1).float().sum() == 0.0:
    #                     rendered_image[k, x_tile * 512:(x_tile * 512 + 512), y_tile * 512:(y_tile * 512 + 512)] = image_sub[:, :]
    #                     break
    #                 if gid == (pc.get_xyz.shape[0] - 1):
    #                     rendered_image[k, x_tile * 512:(x_tile * 512 + 512), y_tile * 512:(y_tile * 512 + 512)] = image_sub[:, :]
    #                     break


    # 4X4 tiles per iter
    # for k in range(num_psf):
    #     for x_tile in range(4):
    #         for y_tile in range(4):
    #             for gid in range(pc.get_xyz.shape[0]):
    #                 index = indices[gid]
    #                 xyz = pc.get_xyz[index, :]
    #                 covariance_2D = torch.zeros((2, 2), device="cuda")
    #                 covariance_2D[0, 0] = symm[index, 0]
    #                 covariance_2D[0, 1] = symm[index, 1]
    #                 covariance_2D[1, 0] = symm[index, 1]
    #                 covariance_2D[1, 1] = symm[index, 3]
    #                 opacity = pc.get_opacity[index]
    #                 grey = pc.get_grey[index]
    #                 dist = xy_stack[x_tile*128:(x_tile*128+128), y_tile*128:(y_tile*128+128)] - xyz[:2]
    #                 before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
    #                     (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
    #                                              dist.unsqueeze(-1))).squeeze()
    #                 after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
    #                                       PSF[k, :, :].unsqueeze(0).unsqueeze(0),
    #                                       padding=17)
    #                 rendered_image[k, x_tile*128:(x_tile*128+128), y_tile*128:(y_tile*128+128)] = rendered_image[k, x_tile*128:(x_tile*128+128), y_tile*128:(y_tile*128+128)] + F.interpolate(
    #                     (summation[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)] < 1).float().unsqueeze(0).unsqueeze(0), size=(128, 128),
    #                     mode='nearest').squeeze() * F.interpolate(mask[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4), index].float().unsqueeze(0).unsqueeze(0),
    #                                                               size=(128, 128),
    #                                                               mode='nearest').squeeze() * F.interpolate(
    #                     product[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)].unsqueeze(0).unsqueeze(0), size=(128, 128),
    #                     mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #                 summation[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)] = summation[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)] + opacity * mask[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4), index]
    #                 product[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)] = product[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4)] * (1 - opacity + opacity * (~ mask[x_tile*4:(x_tile*4+4), y_tile*4:(y_tile*4+4), index]))
    #

    # 16X16 tiles per iter
    # for k in range(num_psf):
    #     for gid in range(pc.get_xyz.shape[0]):
    #         index = indices[gid]
    #         xyz = pc.get_xyz[index, :]
    #         covariance_2D = torch.zeros((2, 2), device="cuda")
    #         covariance_2D[0, 0] = symm[index, 0]
    #         covariance_2D[0, 1] = symm[index, 1]
    #         covariance_2D[1, 0] = symm[index, 1]
    #         covariance_2D[1, 1] = symm[index, 3]
    #         opacity = pc.get_opacity[index]
    #         grey = pc.get_grey[index]
    #         dist = xy_stack - xyz[:2]
    #         before_conv = (1 / torch.sqrt(torch.det(covariance_2D))) * torch.exp(
    #             (- 1 / 2) * torch.matmul(torch.matmul(dist.unsqueeze(-2), torch.linalg.pinv(covariance_2D)),
    #                                      dist.unsqueeze(-1))).squeeze()
    #         after_conv = F.conv2d(grey * before_conv.unsqueeze(0).unsqueeze(0),
    #                               PSF[k, :, :].unsqueeze(0).unsqueeze(0),
    #                               padding=17)
    #         rendered_image[k, :, :] = rendered_image[k, :, :] + F.interpolate((summation < 1).float().unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze() * F.interpolate(mask[:, :, index].float().unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze() * F.interpolate(product.unsqueeze(0).unsqueeze(0), size=(512, 512), mode='nearest').squeeze() * opacity * after_conv.squeeze()
    #         summation = summation + opacity * mask[:, :, index]
    #         product = product * (1 - opacity + opacity * (~ mask[:, :, index]))


    # with multiprocessing.Pool(processes=4) as pool:
    #     rendered_tiles = []
    #     for k in range(num_psf):
    #         for i in range(num_tiles):
    #             for j in range(num_tiles):
    #                 rendered_tiles.append(pool.apply_async(process_tile, (k, i, j, tile_size, pc, symm, xy_stack, indices, PSF)))


    # for k in range(num_psf):
    #     for i in range(num_tiles):
    #         for j in range(num_tiles):
    #             t = threading.Thread(target=process_tile, args=(k, i, j, tile_size, pc, symm, xy_stack, indices, PSF, rendered_image))
    #             t.start()
    #             threads.append(t)
    #
    # for t in threads:
    #     t.join()


    # for k in range(num_psf):
    #     for i in range(num_tiles):
    #         for j in range(num_tiles):
    #             p = Process(target=process_tile, args=(k, i, j, tile_size, pc, symm, xy_stack, indices, PSF, rendered_image))
    #             processes.append(p)
    #             p.start()
    #
    # for p in processes:
    #     p.join()

    # print("flag")
    # print(flag.sum())
    # print("cnt")
    # print(cnt)
    # print("cnt_mask")
    # print(cnt_mask)
    return rendered_image
