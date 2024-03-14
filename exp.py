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
a = torch.ones((2,2))
print(a.sum())