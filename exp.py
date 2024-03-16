# import multiprocessing
#
#
# def process_task(number):
#     result = number * 2
#     print(f"处理数字 {number}，结果为 {result}")
#
#
# if __name__ == "__main__":
#     numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#
#     with multiprocessing.Pool(processes=10) as pool:
#         pool.map(process_task, numbers)
import torch
from scipy.io import loadmat
import torch.nn.functional as F
from PIL import Image
import numpy as np
# a = torch.ones((2,2))
# print(a.sum())
# a = torch.rand((32, 32))
# b = torch.zeros((35, 35))
# b[17, 17] = 1
# print(a.shape)
# print(a)
# c = F.conv2d(a.unsqueeze(0).unsqueeze(0), b.unsqueeze(0).unsqueeze(0), padding=17)
# print(c.shape)
# print(c)
# gt_image = torch.zeros((81, 512, 512), device="cuda")
# # PSF = torch.zeros((81, 35, 35), device="cuda")
# PSF = torch.zeros((81, 135, 135), device="cuda")
#
# for i in range(PSF.shape[0]):
#     psf_data = loadmat('G:/imaging/metasensor/training_data/psf_{}.mat'.format(i + 1))
#     # psf = psf_data['psf'][50:85, 50:85]
#     psf = psf_data['psf'][:, :]
#     PSF[i, :, :] = torch.tensor(psf, device="cuda")
#
#     abbe_image_data = loadmat('G:/imaging/metasensor/training_data/abbe_image_{}.mat'.format(i + 1))
#     abbe_image = abbe_image_data['abbe_image']
#     # print('abbe_image')
#     # print(abbe_image)
#     gt_image[i, :, :] = torch.tensor(abbe_image, device="cuda")
#     # print('gt_image')
#     # print(gt_image)
#
# min_vals, _ = torch.min(torch.min(gt_image, dim=1, keepdim=True)[0], dim=2, keepdim=True)
# max_vals, _ = torch.max(torch.max(gt_image, dim=1, keepdim=True)[0], dim=2, keepdim=True)
# print(min_vals.shape)
# print(max_vals.shape)
# gt_image = (gt_image - min_vals) / (max_vals - min_vals)
#
# # a = torch.rand((4,4))
# # print(torch.min(a).shape)
# # print(torch.min(a, dim=1)[0].shape)
# # print(torch.min(a, dim=1, keepdim=True)[0].shape)
# # print(torch.min(a, dim=1))
#
# for i in range(PSF.shape[0]):
#     im = PSF[i, :, :]
#     im_np = im.cpu().numpy()
#     im_normalized = (im_np - im_np.min()) / (im_np.max() - im_np.min())
#     im_normalized = im_normalized * 255
#     im_uint8 = im_normalized.astype(np.uint8)
#     im_save = Image.fromarray(im_uint8, mode='L')
#     im_save.save('./results/PSF_full_{}.png'.format(i))
#
#     im = gt_image[i, :, :]
#     im_np = im.cpu().numpy()
#     im_normalized = (im_np - im_np.min()) / (im_np.max() - im_np.min())
#     im_normalized = im_normalized * 255
#     im_uint8 = im_normalized.astype(np.uint8)
#     im_save = Image.fromarray(im_uint8, mode='L')
#     im_save.save('./results/gt_image_{}.png'.format(i))
#
#     # im = gt_image[i, :, :]
#     # # print(im)
#     # im_np = im.cpu().numpy()
#     # im_np = im_np.astype(np.uint8)
#     # im_save = Image.fromarray(im_np, mode='L')
#     # im_save.save('./results/gt_image_origin_{}.png'.format(i))
# a = torch.tensor([1,2,3])
# b = torch.tensor([[1,1], [1,1]])
# print(a.unsqueeze(0).shape)
# print(a.expand(2,2,-1)[1,0,:])
# a = torch.tensor([[1,2], [3,4]])
# print(a)
# print(a.repeat(3,3))
import torch
import torch.nn.functional as F

# # 创建一个2x2的原始矩阵
# original_matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加两个维度以匹配上采样函数的要求
#
# # 定义目标尺寸
# target_size = (4, 4)
#
# 使用最近邻插值进行上采样
# expanded_matrix = F.interpolate(original_matrix, size=target_size, mode='nearest')

# # 输出结果
# print(expanded_matrix.squeeze())
a = torch.rand(2,2)>3
print(a)
print(a.float().sum() == 0.0)
# print(~a)
# print(a)