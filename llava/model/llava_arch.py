#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import MiniBatchKMeans
# import faiss
from sklearn import random_projection
import scipy.sparse
# import faiss
# # import cupy as cp
# from cuml.cluster import KMeans
# from sklearn.cluster import KMeans



# def lsh_deduplicate(tensor, d):
#     tensor = tensor.to(torch.float32)  # PyTorch: 转换 float16 -> float32
#     tensor_np = tensor.cpu().numpy()  # 转换为 NumPy 数组

#     n, dim = tensor_np.shape

#     # 构建 LSH 索引
#     nbits = int(np.log2(n))
#     index = faiss.IndexLSH(dim, nbits)
#     index.train(tensor_np)
#     index.add(tensor_np)

#     # 进行最近邻搜索
#     _, indices = index.search(tensor_np, 2)
#     unique_indices = list(set(indices[:, 0]))[:d]  # 选择 d 个去重样本

#     result = torch.tensor(tensor_np[unique_indices], dtype=torch.float16)  # 转换回 PyTorch float16
#     return result




def compute_entropy(tensor):
    """
    快速计算信息熵
    :param tensor: 输入张量，形状为 (num_tokens, feature_dim)
    :return: 每个 token 的信息熵，形状为 (num_tokens,)
    """
    # 取绝对值并归一化，得到概率分布
    prob_dist = F.normalize(tensor.abs(), p=1, dim=1)
    
    # 计算信息熵：entropy = -sum(p * log(p))
    entropy_values = -torch.sum(prob_dist * torch.log(prob_dist + 1e-10), dim=1)
    return entropy_values

# def RLP9(image_feature, num_tokens_per_frame=196, merging_ratio=0.7, keep_removed_ratio=0.1):
#     # 计算帧数
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
    
#     # 计算需要保留的 token 比例
#     merging_ratio = 1 - merging_ratio  # 例如，merging_ratio=0.7 表示保留 30% 的 token

#     # 归一化 tokens（批量归一化）
#     normed_tokens = F.normalize(image_feature, p=2, dim=1)

#     # 重新 reshape 为 (batch, num_tokens, feature_dim)，便于批量计算
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 获取相邻帧
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

#     # 计算余弦相似度（使用批量矩阵乘法）
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 构建新的 token 列表
#     modified_image_feature = []

#     # 遍历每一帧
#     for i in range(num_frames):
#         if i == 0:
#             # 第一帧直接保留所有 tokens
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])
#         else:
#             # 对于后续帧，筛选 tokens
#             # 1. 保留低相似度的 tokens
#             low_sim_indices = similarities[i - 1].topk(num_tokens_to_keep, largest=False).indices
#             low_sim_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame][low_sim_indices]
            
#             # 2. 从裁剪掉的 tokens 中筛选信息熵最高的 10%
#             removed_indices = torch.arange(num_tokens_per_frame, device=image_feature.device)
#             removed_indices = removed_indices[~torch.isin(removed_indices, low_sim_indices)]
#             removed_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame][removed_indices]
            
#             # 计算信息熵
#             entropy_values = compute_entropy(removed_tokens)
            
#             # 计算需要保留的 token 数量
#             num_removed_to_keep = int(keep_removed_ratio * len(removed_indices))
            
#             # 选择信息熵最高的 tokens
#             if num_removed_to_keep > 0:
#                 high_entropy_indices = entropy_values.topk(num_removed_to_keep, largest=True).indices
#                 high_entropy_tokens = removed_tokens[high_entropy_indices]
                
#                 # 合并低相似度和高信息熵的 tokens
#                 combined_tokens = torch.cat([low_sim_tokens, high_entropy_tokens], dim=0)
                
#                 # 按照原始顺序排序
#                 all_indices = torch.cat([low_sim_indices, removed_indices[high_entropy_indices]])
#                 sorted_indices = torch.sort(all_indices).indices
#                 combined_tokens = combined_tokens[sorted_indices]
                
#                 # 添加到结果中
#                 modified_image_feature.append(combined_tokens)
#             else:
#                 # 如果没有需要保留的高信息熵 tokens，直接添加低相似度 tokens
#                 modified_image_feature.append(low_sim_tokens)

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
    
#     # 返回合并后的 tokens
#     return combined_tokens











# def RLP8(image_feature, num_tokens_per_frame=196, merging_ratio=0.7, target_dim=100):
#     """
#     对图像特征进行降维和 token 筛选，所有计算在 GPU 上完成。
    
#     参数:
#         image_feature (torch.Tensor): 输入图像特征，形状为 (num_tokens, feature_dim)，位于 GPU 上。
#         num_tokens_per_frame (int): 每帧的 token 数量。
#         merging_ratio (float): 合并比例，保留 1 - merging_ratio 的 token。
#         target_dim (int): 随机投影的目标维度。
    
#     返回:
#         combined_tokens (torch.Tensor): 筛选后的 tokens，位于 GPU 上。
#     """
#     device = image_feature.device  # 获取输入数据的设备（GPU）
#     num_frames = image_feature.shape  [0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # 手动生成稀疏随机矩阵
#     feature_dim = image_feature.shape  [1]  # 原始特征维度
#     density = 1.0 / np.sqrt(feature_dim)  # 稀疏矩阵的密度
#     random_matrix = scipy.sparse.rand(feature_dim, target_dim, density=density, format='csr', random_state=42)
#     random_matrix.data = np.random.choice([-1, 1], size=random_matrix.data.shape)  # 将非零元素设置为 -1 或 1
#     random_matrix = torch.tensor(random_matrix.toarray(), dtype=torch.float16).to(device)  # 转换为密集矩阵并移动到 GPU，且转换为 float16

#     # 在 GPU 上进行降维计算
#     image_feature_reduced = torch.matmul(image_feature, random_matrix)  # 矩阵乘法

#     # 归一化 tokens（批量归一化）
#     normed_tokens = torch.nn.functional.normalize(image_feature_reduced, p=2, dim=1)

#     # 重新 reshape 为 (batch, num_tokens, feature_dim)，便于批量计算
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 获取相邻帧
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

#     # 计算余弦相似度（使用批量矩阵乘法）
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 选择低相似度的 token（topk 选取最不相似的部分）
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 选取不相似的 token

#     # 构建新的 token 列表
#     modified_image_feature = []

#     # 保留所有帧
#     for i in range(num_frames - 1):
#         if i < 1:
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])  # 直接保留 frame1
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分
#         else:
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens



# from kmeans_pytorch import kmeans  # 使用 GPU 加速的 K-Means 实现


# def kmeans_gpu(X, num_clusters, max_iters=10, tol=1e-4):
#     """
#     使用 PyTorch 在 GPU 上运行 K-Means 聚类
#     """
#     X = X.float()  # 确保数据是 float32（FP16 精度可能不足）
#     device = X.device

#     # 初始化聚类中心（随机选择数据点）
#     indices = torch.randperm(X.shape[0], device=device)[:num_clusters]
#     centroids = X[indices]

#     for _ in range(max_iters):
#         # 计算距离
#         distances = torch.cdist(X, centroids)  # 计算 L2 距离
#         labels = torch.argmin(distances, dim=1)

#         # 计算新聚类中心
#         new_centroids = torch.stack([X[labels == i].mean(dim=0) for i in range(num_clusters)])

#         # 检查收敛
#         if torch.norm(new_centroids - centroids) < tol:
#             break

#         centroids = new_centroids

#     return centroids.half()  # 转回 FP16

# def dynamic_clustering_on_gpu(tokens, max_clusters=10):
#     """
#     在 GPU 上对 tokens 进行动态聚类（支持 FP16）。
#     """
#     num_tokens = tokens.shape[0]
#     if num_tokens == 0:
#         return tokens

#     num_clusters = min(num_tokens // 10, max_clusters)
#     if num_clusters < 1:
#         return tokens

#     return kmeans_gpu(tokens, num_clusters)

# def RLP7(image_feature, num_tokens_per_frame=196, merging_ratio=0.7, max_clusters=10):
#     """
#     优化图像特征，减少冗余信息并保留关键特征，同时对裁剪掉的 tokens 基于动态聚类进行快速融合（支持 FP16）。

#     参数:
#     - image_feature: 输入的特征张量，形状为 [总token数, 特征维度]，数据类型为 FP16。
#     - num_tokens_per_frame: 每帧的token数量，默认为196。
#     - merging_ratio: 合并比例，表示需要保留的token比例，默认为0.7。
#     - max_clusters: 最大聚类数量，默认为10。

#     返回:
#     - combined_tokens: 优化后的特征张量，数据类型为 FP16。
#     """
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # 确保数据在 GPU 上且为 FP16
#     image_feature = image_feature.to('cuda').half()

#     # 归一化 tokens（批量归一化）
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

#     # 重新 reshape 为 (batch, num_tokens, feature_dim)，便于批量计算
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 获取相邻帧
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

#     # 计算余弦相似度（使用批量矩阵乘法）
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 选择低相似度的 token（topk 选取最不相似的部分）
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 选取不相似的 token

#     # 构建新的 token 列表
#     modified_image_feature = []

#     # 保留所有帧
#     for i in range(num_frames - 1):
#         if i < 1:
#             # 直接保留第一帧的所有token
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])
#             # 保留第二帧中相异度较高的部分
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)
#         else:
#             # 仅保留第二帧中相异度较高的部分
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)

#         # 对裁剪掉的 tokens 基于动态聚类进行快速融合
#         frame2_tokens_all = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
#         frame2_tokens_discarded_indices = torch.tensor([idx for idx in range(num_tokens_per_frame) if idx not in tokens_to_keep[i]], device=image_feature.device)
#         frame2_tokens_discarded = frame2_tokens_all[frame2_tokens_discarded_indices]

#         # 对裁剪掉的 tokens 进行动态聚类
#         if len(frame2_tokens_discarded) > 0:
#             cluster_centers = dynamic_clustering_on_gpu(frame2_tokens_discarded, max_clusters)
#             if isinstance(cluster_centers, tuple):  # 如果返回的是元组，只取第一个元素
#                 cluster_centers = cluster_centers[0]
#             # print(cluster_centers.shape)
#             modified_image_feature.append(cluster_centers)

#     # 确保所有张量都在 GPU 上
#     modified_image_feature = [tensor.to('cuda') if isinstance(tensor, torch.Tensor) else tensor for tensor in modified_image_feature]

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens

# def RLP6(image_feature, num_tokens_per_frame=196, merging_ratio=0.7): 
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # 归一化 tokens（批量归一化）
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

#     # 重新 reshape 为 (num_frames, num_tokens_per_frame, feature_dim)
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 计算余弦相似度（使用批量矩阵乘法）
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
#         # 保存所有相似度值到文本文件
#     with open("all_similarities.txt", "a") as f:  # "a" 代表 append 模式
#         f.write("New Run:\n")  # 每次运行新增一组数据
#         for i, frame_sims in enumerate(similarities):
#             f.write(f"Frame {i+1}:\n")
#             f.write(" ".join(map(str, frame_sims.tolist())) + "\n")

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 选择低相似度的 token（topk 选取最不相似的部分）
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # [num_frames-1, num_tokens_to_keep]

#     # # 计算保留 tokens 的最大相似度
#     max_similarity_kept = similarities.gather(dim=1, index=tokens_to_keep).max().item()
#     

#     # 重新 reshape 原始 `image_feature` 以匹配 shape
#     image_feature = image_feature.view(num_frames, num_tokens_per_frame, -1)

#     # 构建新的 token 列表
#     modified_image_feature = [image_feature[0]]  # 直接保留第一帧

#     # 逐帧保留低相似度的 token
#     for i in range(num_frames - 1):
#         selected_tokens = torch.gather(image_feature[i + 1], dim=0, index=tokens_to_keep[i].unsqueeze(-1).expand(-1, image_feature.shape[-1]))
#         modified_image_feature.append(selected_tokens)

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens

# def RLP5(image_feature, num_tokens_per_frame=196, merging_ratio=0.7):
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # 归一化 tokens（批量归一化）
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

#     # 重新 reshape 为 (batch, num_tokens, feature_dim)，便于批量计算
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 获取相邻帧
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

#     # 计算余弦相似度（使用批量矩阵乘法）
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
#     # print(similarities.shape)
#     # print(similarities.size)

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 选择低相似度的 token（topk 选取最不相似的部分）
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 选取不相似的 token

#     # 构建新的 token 列表
#     modified_image_feature = []

#     # 保留所有帧
#     for i in range(num_frames - 1):
#         if i < 1:
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])  # 直接保留 frame1
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分
#         else :
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens

# def RLP4(new_input_embeds, image_feature, new_labels, merging_ratio=0.7):
#     num_frames = 31
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     SYS_LENGTH = 14
#     IMAGE_TOKENS = image_feature.shape[0]  # 图像 tokens 数量
#     ATTENTION_RANK = IMAGE_TOKENS - 196  #
#     first_IMAGE_TOKENS = 196
#     every_frames = int(ATTENTION_RANK/num_frames)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     all_tokens = new_input_embeds[0]  # [Total_Tokens, Hidden_Dim]

#     # 拆分图像 tokens 和文本 tokens
#     image_tokens = all_tokens[SYS_LENGTH:SYS_LENGTH + IMAGE_TOKENS]
#     # image_tokens1 = image_tokens[:first_IMAGE_TOKENS]  # 第一帧 tokens 全部保留
#     image_tokens2 = image_tokens[first_IMAGE_TOKENS:]  # 其余 31 帧 tokens
    
#     text_tokens = torch.cat([all_tokens[:SYS_LENGTH],
#                              all_tokens[SYS_LENGTH + IMAGE_TOKENS:]], dim=0)

#     # L2 归一化
#     image_tokens2 = F.normalize(image_tokens2, p=2, dim=1)
#     text_tokens = F.normalize(text_tokens, p=2, dim=1)

#     # 逐帧计算相似度
#     # selected_image_tokens = [image_tokens1]  # 先保留第一帧
#     temp_indexs = []
#     num_tokens_to_keep = int(every_frames * merging_ratio)

#     for i in range(num_frames):
#         frame_tokens = image_tokens2[i * every_frames:(i + 1) * every_frames]
#         similarity_matrix = frame_tokens @ text_tokens.T  # 计算相似度
#         sum_sim = similarity_matrix.sum(dim=1)  # 计算总相似度

#         # 计算每帧要保留的 tokens 数量
        
#         topk_indices = sum_sim.topk(num_tokens_to_keep).indices +  SYS_LENGTH + first_IMAGE_TOKENS # 选取高相似度 tokens
#         # selected_image_tokens.append(frame_tokens[topk_indices])
#         temp_indexs.append(topk_indices)

#     # 合并保留的 tokens
#     # selected_image_tokens = torch.cat(selected_image_tokens, dim=0)
#     temp_indexs = torch.cat(temp_indexs, dim=0)  # 将列表转换为 Tensor
#     # print(f"image_feature.shape[0]:{image_feature.shape[0]}")
#     # print(f"image_tokens2:{image_tokens2.shape}")
#     # print(f"text_tokens:{text_tokens.shape}")
#     # print(f"every_frames:{every_frames}")
#     # print(f"num_tokens_to_keep:{num_tokens_to_keep}")
#     # print(f"temp_indexs:{temp_indexs.shape}")

#     # 组合索引
#     keep_indexs = torch.cat((
#         torch.arange(SYS_LENGTH + first_IMAGE_TOKENS, device=device),  # 保留系统 tokens
#         temp_indexs,  # 
#         torch.arange(SYS_LENGTH + IMAGE_TOKENS, all_tokens.shape[0], device=device)  # 文本 tokens
#     ))

#     # print(f"keep_indexs:{keep_indexs.shape}")
#     # 排序索引
#     keep_indexs = keep_indexs.sort().values

#     # 更新 input_embeds 和 labels
#     new_input_embeds[0] = new_input_embeds[0][keep_indexs, :]
#     new_labels[0] = new_labels[0][keep_indexs]

#     return new_input_embeds, new_labels

# def RLP3(image_feature, num_tokens_per_frame=196, merging_ratio=0.7):
#     """
#     第一帧 tokens 全部保留，后续帧使用 L2 范数筛选。
#     :param image_feature: ViT 提取的图像特征，shape=(num_frames * num_tokens_per_frame, feature_dim)
#     :param num_tokens_per_frame: 每帧的 token 数量
#     :param merging_ratio: 需要合并的比例 (0.7 表示仅保留 30% 的 token)
#     :return: 筛选后的 tokens
#     """
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     keep_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # Step 1: 归一化 tokens
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

#     # Step 2: 重新 reshape 为 (num_frames, num_tokens_per_frame, feature_dim)
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # Step 3: 计算 L2 范数
#     l2_norms = torch.norm(normed_tokens, p=2, dim=-1)  # 计算每个 token 的 L2 范数
#     num_tokens_to_keep = int(keep_ratio * num_tokens_per_frame)  # 计算后续帧需要保留的 token 数量

#     # Step 4: 构建新的 token 列表
#     modified_image_feature = []

#     # 第一帧全部保留
#     modified_image_feature.append(image_feature[:num_tokens_per_frame])

#     # 后续帧进行 L2 范数筛选
#     for i in range(1, num_frames):
#         frame_tokens = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]  # 当前帧所有 tokens
#         tokens_to_keep = l2_norms[i].topk(num_tokens_to_keep, largest=True).indices  # 选取 L2 最大的 tokens
#         frame_selected = frame_tokens[tokens_to_keep]  # 选择保留的 tokens
#         modified_image_feature.append(frame_selected)

#     # Step 5: 合并所有筛选后的 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)

#     return combined_tokens   

# def RLP2(image_feature, num_tokens_per_frame=196, merging_ratio=0.7): 
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  

#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     frame1_tokens = normed_tokens[:-1]
#     frame2_tokens = normed_tokens[1:]

#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  

#     modified_image_feature = []

#     for i in range(num_frames - 1):
#         if i < 1:
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])

#         frame2_all_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
#         frame2_selected_tokens = frame2_all_tokens[tokens_to_keep[i]]

#         # 获取被裁剪掉的 tokens
#         discarded_tokens = frame2_all_tokens[~tokens_to_keep[i]]

#         if discarded_tokens.numel() > 0:
#             # 计算注意力权重
#             attn_weights = torch.nn.functional.softmax(similarities[i][~tokens_to_keep[i]], dim=-1).unsqueeze(-1)
            
#             # 计算加权均值
#             weighted_token = (attn_weights * discarded_tokens).sum(dim=0, keepdim=True)
            
#             # 添加到保留 tokens
#             frame2_selected_tokens = torch.cat([frame2_selected_tokens, weighted_token], dim=0)

#         modified_image_feature.append(frame2_selected_tokens)

#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens

# def dycole_ttm_attention_distillation(image_feature, num_tokens_per_frame=196, merging_ratio=0.7, refine_ratio=0.1):
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算保留的比例
    
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     frame1_tokens = normed_tokens[:-1]
#     frame2_tokens = normed_tokens[1:]

#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 选择最不相似的 tokens 保留

#     modified_image_feature = []
    
#     for i in range(num_frames - 1):
#         if i < 1:
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])
        
#         frame2_all_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
#         frame2_selected_tokens = frame2_all_tokens[tokens_to_keep[i]]
#         # print(f"frame2_selected_tokens:{frame2_selected_tokens.shape}")

#         # 获取被裁剪掉的 tokens 及其索引
#         all_indices = torch.arange(num_tokens_per_frame, device=image_feature.device)
#         discarded_mask = torch.ones(num_tokens_per_frame, dtype=torch.bool, device=image_feature.device)
#         discarded_mask[tokens_to_keep[i]] = False
#         discarded_indices = all_indices[discarded_mask]
#         discarded_tokens = frame2_all_tokens[discarded_mask]
        
#         if discarded_tokens.numel() > 0:
#             # 计算裁剪 token 与保留 token 的相似度
#             retained_normed_tokens = normed_tokens[i + 1][tokens_to_keep[i]]
#             discarded_normed_tokens = normed_tokens[i + 1][discarded_mask]
#             discarded_similarities = torch.matmul(discarded_normed_tokens, retained_normed_tokens.T)
            
#             # 选择相似度最高的 refine_ratio 部分 token 进行保留
#             num_refined_tokens = int(refine_ratio * discarded_tokens.shape[0])
#             # num_refined_tokens = 10
#             refined_indices = discarded_similarities.max(dim=1).indices.topk(num_refined_tokens, largest=True).indices
#             refined_tokens = discarded_tokens[refined_indices]
#             refined_original_indices = discarded_indices[refined_indices]
#             # print(f"refined_tokens:{refined_tokens.shape}")
            
            
#             # 重新按原始索引排序
#             combined_indices = torch.cat([tokens_to_keep[i], refined_original_indices])
#             combined_tokens = torch.cat([frame2_selected_tokens, refined_tokens], dim=0)
#             sorted_indices = combined_indices.argsort()
#             frame2_selected_tokens = combined_tokens[sorted_indices]
        
#         modified_image_feature.append(frame2_selected_tokens)
    
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens


@torch.jit.script  # JIT 编译优化
def RLP1(image_feature: torch.Tensor, 
                                      num_tokens_per_frame: int = 196, 
                                      merging_ratio: float = 0.7, 
                                      refine_ratio: float = 0.1) -> torch.Tensor:
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio  # 计算保留比例
    
    normed_tokens = torch.nn.functional.normalize(image_feature, p=2.0, dim=1)
    normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)
    
    frame1_tokens = normed_tokens[:-1]
    frame2_tokens = normed_tokens[1:]
    
    # 使用 einsum 计算相似度，提高计算效率
    similarities = torch.einsum('bik,bjk->bi', frame1_tokens, frame2_tokens)
    num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
    tokens_to_keep = torch.topk(similarities, num_tokens_to_keep, largest=False).indices  # 选择最不相似的 tokens 保留
    
    modified_image_feature = []
    
    for i in range(num_frames - 1):
        if i < 1:
            modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])
        
        frame2_all_tokens = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]
        frame2_selected_tokens = torch.index_select(frame2_all_tokens, 0, tokens_to_keep[i])
        
        # # 获取被裁剪掉的 tokens 及其索引
        # all_indices = torch.arange(num_tokens_per_frame, device=image_feature.device)
        # discarded_mask = torch.ones(num_tokens_per_frame, dtype=torch.bool, device=image_feature.device)
        # discarded_mask[tokens_to_keep[i]] = False
        # discarded_indices = all_indices[discarded_mask]
        # discarded_tokens = frame2_all_tokens[discarded_mask]
        
        # if discarded_tokens.numel() > 0:
        #     retained_normed_tokens = normed_tokens[i + 1][tokens_to_keep[i]]
        #     discarded_normed_tokens = normed_tokens[i + 1][discarded_mask]
        #     discarded_similarities = torch.matmul(discarded_normed_tokens, retained_normed_tokens.T)
            
        #     num_refined_tokens = int(refine_ratio * discarded_tokens.shape[0])
        #     refined_indices = torch.topk(discarded_similarities.max(dim=1).indices, num_refined_tokens, largest=True).indices
        #     refined_tokens = torch.index_select(discarded_tokens, 0, refined_indices)
        #     refined_original_indices = torch.index_select(discarded_indices, 0, refined_indices)
            
        #     # 重新按原始索引排序
        #     combined_indices = torch.cat([tokens_to_keep[i], refined_original_indices])
        #     combined_tokens = torch.cat([frame2_selected_tokens, refined_tokens], dim=0)
        #     sorted_indices = torch.topk(combined_indices, combined_indices.shape[0], largest=False).indices
        #     frame2_selected_tokens = torch.index_select(combined_tokens, 0, sorted_indices)
        
        modified_image_feature.append(frame2_selected_tokens)
    
    return torch.cat(modified_image_feature, dim=0)


def RLP(image_feature, num_tokens_per_frame=196, merging_ratio=0.7):
    num_frames = image_feature.shape[0] // num_tokens_per_frame
    merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

    # 归一化 tokens（批量归一化）
    normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

    # 重新 reshape 为 (num_frames, num_tokens, feature_dim)，便于批量计算
    normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

    # 获取相邻帧
    frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
    frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

    # 计算余弦相似度（使用批量矩阵乘法）
    similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)

    # 计算每帧要保留的 token 数量
    num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)
    num_tokens_to_restore = int((num_tokens_per_frame - num_tokens_to_keep)*0.1)  # 恢复的 token 数量
    # print(num_tokens_to_keep)
    # print(num_tokens_to_restore)

    # 选择低相似度的 token（topk 选取最不相似的部分）
    tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 初步筛选的 tokens
    

    # 构建新的 token 列表
    modified_image_feature = []

    for i in range(num_frames - 1):
        if i < 1:
            frame1_selected = image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame]  
            modified_image_feature.append(frame1_selected)  # 第一帧完整保留
        

        frame2_all = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame]  

        # 先保留第一步筛选出的 tokens
        frame2_selected = frame2_all[tokens_to_keep[i]]
        modified_image_feature.append(frame2_selected)

        # 获取被裁剪掉的 tokens
        mask = torch.ones(num_tokens_per_frame, dtype=torch.bool, device=image_feature.device)
        mask[tokens_to_keep[i]] = False  # 将保留的 tokens 设为 False，剩下的为裁剪部分
        frame2_cropped = frame2_all[mask]  # 剩余裁剪 tokens

        # 计算裁剪 tokens 与第一步保留 tokens 的相似度
        normed_frame2_selected = torch.nn.functional.normalize(frame2_selected, p=2, dim=1)
        normed_frame2_cropped = torch.nn.functional.normalize(frame2_cropped, p=2, dim=1)
        
        # cropped_similarities = torch.mm(normed_frame2_cropped, normed_frame2_selected.T).mean(dim=1) 
        cropped_similarities = torch.mm(normed_frame2_cropped, normed_frame2_selected.T).max(dim=1).values

        # 选择裁剪部分中相似度高的 tokens 进行恢复
        
        restore_indices = cropped_similarities.topk(num_tokens_to_restore, largest=True).indices
        restored_tokens = frame2_cropped[restore_indices]

        # 组合最终保留的 tokens
        final_tokens = torch.cat([frame2_selected, restored_tokens], dim=0)

        # 按照原始索引顺序排列
        restore_original_indices = torch.cat([tokens_to_keep[i], mask.nonzero().squeeze(1)[restore_indices]])
        sorted_indices = restore_original_indices.sort().indices
        final_tokens = final_tokens[sorted_indices]

        modified_image_feature.append(final_tokens)

    # 合并所有 tokens
    combined_tokens = torch.cat(modified_image_feature, dim=0)
    return combined_tokens


# def RLP(image_feature, num_tokens_per_frame=196, merging_ratio=0.7):
#     num_frames = image_feature.shape[0] // num_tokens_per_frame
#     merging_ratio = 1 - merging_ratio  # 计算需要保留的比例

#     # 归一化 tokens（批量归一化）
#     normed_tokens = torch.nn.functional.normalize(image_feature, p=2, dim=1)

#     # 重新 reshape 为 (batch, num_tokens, feature_dim)，便于批量计算
#     normed_tokens = normed_tokens.view(num_frames, num_tokens_per_frame, -1)

#     # 获取相邻帧
#     frame1_tokens = normed_tokens[:-1]  # [num_frames-1, num_tokens, feature_dim]
#     frame2_tokens = normed_tokens[1:]   # [num_frames-1, num_tokens, feature_dim]

#     # 计算余弦相似度（使用批量矩阵乘法）
#     similarities = torch.bmm(frame1_tokens, frame2_tokens.transpose(1, 2)).diagonal(dim1=-2, dim2=-1)
#     # print(similarities.shape)
#     # print(similarities.size)

#     # 计算每帧要保留的 token 数量
#     num_tokens_to_keep = int(merging_ratio * num_tokens_per_frame)

#     # 选择低相似度的 token（topk 选取最不相似的部分）
#     tokens_to_keep = similarities.topk(num_tokens_to_keep, largest=False).indices  # 选取不相似的 token

#     # 构建新的 token 列表
#     modified_image_feature = []

#     # 保留所有帧
#     for i in range(num_frames - 1):
#         if i < 1:
#             modified_image_feature.append(image_feature[i * num_tokens_per_frame: (i + 1) * num_tokens_per_frame])  # 直接保留 frame1
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分
#         else :
#             frame2_tokens_selected = image_feature[(i + 1) * num_tokens_per_frame: (i + 2) * num_tokens_per_frame][tokens_to_keep[i]]
#             modified_image_feature.append(frame2_tokens_selected)  # 仅保留 frame2 中相异度较高的部分

#     # 合并所有 tokens
#     combined_tokens = torch.cat(modified_image_feature, dim=0)
#     return combined_tokens



class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        if height * width != image_feature.shape[1]:
            height = width = int(math.sqrt(image_feature.shape[1]))
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(weight / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)

        image_features = self.get_model().mm_projector(image_features) #torch.Size([32, 729, 3584])
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if self.config.add_faster_video:
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)

        # print(f"images.shape:{images.size}") 

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            # print(split_sizes) [32]

            # print(f"concat_images.shape:{concat_images.shape}")  torch.Size([32, 3, 384, 384])


            encoded_image_features = self.encode_images(concat_images)

            # print(f"encoded_image_features.shape:{encoded_image_features.shape}") torch.Size([32, 729, 3584])

            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)


            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            # print(f"encoded_image_features.shape:{encoded_image_features.size}") 


            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                # [16, 729, 3584]
                # print(f"image_feat.shape:{image_feat.shape}") torch.Size([32, 729, 3584])
                # print(idx) 0
                # print(video_idx_in_batch) [0]
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))
                else:
                    image_features.append(image_feat)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            # print(f"image_features.shape:{image_features[0].shape}") torch.Size([32, 196, 3584])
            

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if self.config.add_faster_video:
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            # print(f"####images.shape:{image_feature.shape}") torch.Size([32, 196, 3584])
                            image_feature = image_feature.flatten(0, 1)
                            # print(f"####image_feature.flatten(0, 1):{image_feature.shape}") torch.Size([6272, 3584])
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = RLP(image_feature, merging_ratio=0.7)
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                                
                            new_image_features.append(image_feature)   
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                #print(cur_input_embeds_no_im[i].size()) [14, 3584]
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    #print(cur_image_features.size()) [3137, 3584]
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()s
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        

        
        
        # new_input_embeds,new_labels = dycole_ttm_text(new_input_embeds, image_feature, new_labels, merging_ratio=0.5)
        # SYS_LENGTH= 14
        # IMAGE_TOKEN_LENGTH = image_feature.shape[0] - 1
        # ATTENTION_RANK = 1024
        
        
        # # print(f"new_input_embeds.shape:{new_input_embeds.shape}")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # 假设 hidden_states 已经加载
        # # hidden_states = hidden_states.to(device)  # 显式移动到 GPU/CPU
        
        # all_tokens = new_input_embeds[0]  # [Total_Tokens, Hidden_Dim]
        
        # # 拆分图像 tokens 和文本 tokens
        # image_tokens = all_tokens[SYS_LENGTH + 196 :SYS_LENGTH + IMAGE_TOKEN_LENGTH]  # [4442, 3854]
        # # text_tokens = all_tokens[SYS_LENGTH + IMAGE_TOKEN_LENGTH:]  # [326, 3854]
        # text_tokens = torch.cat([all_tokens[:SYS_LENGTH],
        #                         all_tokens[SYS_LENGTH + IMAGE_TOKEN_LENGTH:]], dim=0)

        # # Step 1: L2 归一化
        # image_tokens = F.normalize(image_tokens, p=2, dim=1)  # [4442, 3854]
        # text_tokens = F.normalize(text_tokens, p=2, dim=1)    # [326, 3854]

        # # Step 2: 计算点积（等价于余弦相似度）
        # similarity_matrix = image_tokens @ text_tokens.T  # [4442, 326]

        # # Step 3: 计算平均相似度
        # # avg_sim = similarity_matrix.mean(dim=1)  # [4442]，计算已在 device 上
        # # max_sim,_ = similarity_matrix.max(dim=1)  # [4442]，计算已在 device 上
        # sum_sim = similarity_matrix.sum(dim=1)  # [4442]，计算已在 device 上
        # # for x in sum_sim:
        # #     print(x)

        # # fffff
        # # Step 4: 获取 Top-K 索引
        # max_sim_rank_index = sum_sim.topk(ATTENTION_RANK).indices + SYS_LENGTH  # [K]

        # # Step 5: 组合索引（保持所有张量在同一设备上）
        # keep_indexs = torch.cat((
        #     torch.arange(SYS_LENGTH, device=device),  # 保留系统 tokens
        #     max_sim_rank_index,  # 选出的前 K 相关 tokens
        #     torch.arange(SYS_LENGTH + IMAGE_TOKEN_LENGTH, all_tokens.shape[0], device=device)  # 其他 tokens
        # ))

        # # Step 6: 排序索引
        # keep_indexs = keep_indexs.sort().values

        # # Step 7: 筛选 hidden_states
        # new_input_embeds[0] = new_input_embeds[0][keep_indexs, :]
        # new_labels[0] = new_labels[0][keep_indexs]
        # # print(f"new_input_embeds.shape:{new_input_embeds.shape}")
        # # 计算 position_ids

        
        
        


        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        
        
        

        

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # print(f"new_input_embeds:{new_input_embeds.shape}")

            

        
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add

        # print(f"position_ids:{position_ids}")
        # print(f"position_ids.shape:{position_ids.shape}")

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")



        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, None


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
