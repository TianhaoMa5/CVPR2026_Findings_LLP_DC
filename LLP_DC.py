from __future__ import print_function
import random

import time
import argparse
import os
import sys
import itertools

import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, OrderedDict
from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, CurrentValueMeter, WarmupCosineLrScheduler
import tensorboard_logger
import torch.multiprocessing as mp
from ortools.graph import pywrapgraph
from LeNet import LeNet5,LeNet
from resnet import ResNet18CIFAR10,resnet34
from convnet import CIFAR10Model,CustomCNN
import torch
import math
from torchvision import models
from papi import PaPiNet

class Edge:
    """
    保存一条有向边的信息：
    - to: 该边指向的节点
    - rev: 反向边在 adjacency[to] 中的下标
    - capacity: 该边的剩余容量
    - cost: 该边的单位花费
    - flow: 当前流量
    """
    __slots__ = ('to', 'rev', 'capacity', 'cost', 'flow')

    def __init__(self, to, rev, capacity, cost):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost
        self.flow = 0


class MinCostMaxFlow:
    """
    使用 Successive Shortest Path + Bellman-Ford 来求解最小费用最大流。
    """

    def __init__(self, n):
        self.n = n
        self.adjacency = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity, cost):
        """
        在图中添加一条 (u->v) 的边，以及对应的 (v->u) 反向边。
        """
        # 正向边
        self.adjacency[u].append(Edge(v, len(self.adjacency[v]), capacity, cost))
        # 反向边
        self.adjacency[v].append(Edge(u, len(self.adjacency[u]) - 1, 0, -cost))

    def min_cost_max_flow(self, source, sink):
        """
        计算从 source 到 sink 的最大流和对应的最小费用。
        返回 (flow, cost)
        """
        flow, cost = 0, 0
        INF = 10 ** 14

        while True:
            # Bellman-Ford 找最短路
            dist = [math.inf] * self.n
            in_queue = [False] * self.n
            parent_node = [-1] * self.n
            parent_edge = [-1] * self.n

            dist[source] = 0
            queue = [source]
            in_queue[source] = True

            # 寻找从 source 到其它点的最短路(费用最小)
            for q_idx in range(self.n):
                if q_idx >= len(queue):
                    break
                u = queue[q_idx]
                in_queue[u] = False

                for i, edge in enumerate(self.adjacency[u]):
                    if edge.flow < edge.capacity and dist[u] + edge.cost < dist[edge.to]:
                        dist[edge.to] = dist[u] + edge.cost
                        parent_node[edge.to] = u
                        parent_edge[edge.to] = i
                        if not in_queue[edge.to]:
                            queue.append(edge.to)
                            in_queue[edge.to] = True

            if dist[sink] == math.inf:
                # 无法再增广
                break

            # 能找到增广路径，则找可用流量
            augment = INF
            v = sink
            while v != source:
                e_idx = parent_edge[v]
                u = parent_node[v]
                e = self.adjacency[u][e_idx]
                augment = min(augment, e.capacity - e.flow)
                v = u

            # 沿路径发送流并累加花费
            v = sink
            while v != source:
                e_idx = parent_edge[v]
                u = parent_node[v]
                e = self.adjacency[u][e_idx]

                e.flow += augment
                self.adjacency[v][e.rev].flow -= augment
                cost += augment * e.cost

                v = u

            flow += augment

        return flow, cost


def cross_entropy_loss_torch(softmax_matrix, onehot_labels):
    """
    计算交叉熵损失 (PyTorch版本)

    :param softmax_matrix: 预测的softmax矩阵 (batch_size, num_classes)
    :param onehot_labels: 真实的onehot标签矩阵 (batch_size, num_classes)
    :return: 平均交叉熵损失
    """
    # 使用 log_softmax 确保数值稳定性
    log_softmax = torch.log(softmax_matrix + 1e-12)

    # 计算交叉熵
    cross_entropy = -torch.sum(onehot_labels * log_softmax, dim=1)

    # 返回平均损失
    mean_loss = torch.mean(cross_entropy)
    return mean_loss


def solve_optimal_onehot_with_proportions_torch(
    softmax_tensor: torch.Tensor,
    proportions: torch.Tensor,
    bagsize: int,
    n_classes: int,
    epsilon=1e-12,
    cost_scale=10000
):
    """
    使用 OR-Tools 求解最优 One-Hot 矩阵（基于 PyTorch 张量）。

    参数：
    --------
    softmax_tensor : torch.Tensor
        形状为 [bagsize, n_classes] 的张量，表示每行的 softmax 概率。
    proportions : torch.Tensor
        形状为 [n_classes] 的张量，表示每个类别的比例，和为 1。
    bagsize : int
        批量大小。
    n_classes : int
        类别数量。
    epsilon : float, optional
        避免 log(0) 的数值安全阈，默认值为 1e-12。
    cost_scale : int, optional
        用于放大成本的倍数，默认值为 10000。

    返回：
    --------
    best_onehot : torch.Tensor
        形状为 [bagsize, n_classes] 的 One-Hot 矩阵，表示分配结果。
    """
    # 确保 softmax_tensor 和 proportions 为 PyTorch 张量
    assert isinstance(softmax_tensor, torch.Tensor), "softmax_tensor 必须是 PyTorch 张量"
    assert isinstance(proportions, torch.Tensor), "proportions 必须是 PyTorch 张量"

    # 转到 CPU 并转换为 NumPy 数组
    softmax_cpu = softmax_tensor.detach().cpu().numpy()
    proportions_cpu = proportions.detach().cpu().numpy()

    # 检查 proportions 的和是否接近 1
    if not math.isclose(proportions_cpu.sum(), 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("proportions 的和必须接近 1。")

    # 1) 计算每个类别的目标数量
    target_counts = (proportions_cpu * bagsize).astype(int)
    remaining = bagsize - target_counts.sum()

    # 调整以确保分配总和等于 batch_size
    if remaining > 0:
        fractional_parts = proportions_cpu * bagsize - target_counts
        sorted_indices = fractional_parts.argsort()[::-1]
        for idx in sorted_indices[:remaining]:
            target_counts[idx] += 1

    # 2) 定义 OR-Tools 的 SimpleMinCostFlow
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # 节点编号
    S = 0
    T = bagsize + n_classes + 1

    def sample_node(i):
        return i + 1

    def class_node(j):
        return bagsize + 1 + j

    # 3) 添加 S -> 样本 节点
    for i in range(bagsize):
        min_cost_flow.AddArcWithCapacityAndUnitCost(S, sample_node(i), 1, 0)

    # 4) 添加 样本 -> 类别 节点
    for i in range(bagsize):
        for j in range(n_classes):
            p_ij = max(softmax_cpu[i, j], epsilon)
            cost = int(-math.log(p_ij) * cost_scale)
            min_cost_flow.AddArcWithCapacityAndUnitCost(sample_node(i), class_node(j), 1, cost)

    # 5) 添加 类别 -> T 节点
    for j in range(n_classes):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            class_node(j), T, int(target_counts[j]), 0
        )
    # 6) 设置供需
    min_cost_flow.SetNodeSupply(S, bagsize)
    min_cost_flow.SetNodeSupply(T, -bagsize)

    # 7) 求解
    status = min_cost_flow.Solve()
    if status != min_cost_flow.OPTIMAL:
        raise RuntimeError("OR-Tools: 未找到最优解")

    # 8) 构建 One-Hot 矩阵
    best_onehot = torch.zeros((bagsize, n_classes), dtype=torch.int32)
    for i in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Flow(i) > 0:
            start = min_cost_flow.Tail(i)
            end = min_cost_flow.Head(i)
            if 1 <= start <= bagsize and bagsize + 1 <= end <= bagsize + n_classes:
                sample_idx = start - 1
                class_idx = end - (bagsize + 1)
                best_onehot[sample_idx, class_idx] = 1

    return best_onehot


def solve_mcf_once(
        softmax_cpu: "np.ndarray",
        target_counts: "np.ndarray",
        cost_scale=10000,
        epsilon=1e-12,
        noise_scale=0.0,
        seed=None
):
    """
    在 -log(softmax) 的成本基础上添加随机扰动后，使用 OR-Tools 求解一次最小费用最大流。

    参数
    ----
    softmax_cpu : np.ndarray
        形状 [bagsize, n_classes] 的 softmax 概率 (CPU的numpy数组)
    target_counts : np.ndarray
        每个类别需要分配的样本个数，长度为 n_classes
    cost_scale : float
        用于放大成本到整数（OR-Tools需要整型cost）
    epsilon : float
        避免 log(0) 时的数值安全阈
    noise_scale : float
        随机扰动的标准差
    seed : int or None
        随机种子，可固定以便可复现

    返回
    ----
    best_onehot : torch.Tensor
        形状 [bagsize, n_classes] 的 0/1 分配矩阵
    total_neglog : float
        该解对应的 sum of (-log(prob) + noise)，用于计算置信度
    """
    if seed is not None:
        random.seed(seed)

    bagsize, n_classes = softmax_cpu.shape
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # 节点编号
    # S = 0, T = bagsize + n_classes + 1
    # 样本节点 i => 1 + i, 类别节点 j => bagsize+1 + j
    S = 0
    T = bagsize + n_classes + 1

    def sample_node(i):
        return 1 + i

    def class_node(j):
        return bagsize + 1 + j

    # ============= (1) 构造边 S->样本、样本->类别、类别->T =============
    # S->样本
    for i in range(bagsize):
        min_cost_flow.AddArcWithCapacityAndUnitCost(S, sample_node(i), 1, 0)

    # 样本->类别
    for i in range(bagsize):
        for j in range(n_classes):
            p_ij = max(softmax_cpu[i, j], epsilon)
            base_cost = -math.log(p_ij)  # 原始 -log(prob)

            # 叠加随机扰动
            if noise_scale > 0:
                noise = random.gauss(0, noise_scale)
                final_cost = base_cost + noise
                # 如担心出现负值，可再做 clamp, e.g. final_cost = max(final_cost, 0.0)
            else:
                final_cost = base_cost

            arc_cost = int(final_cost * cost_scale)  # 转为整数
            min_cost_flow.AddArcWithCapacityAndUnitCost(sample_node(i), class_node(j), 1, arc_cost)

    # 类别->T
    for j in range(n_classes):
        min_cost_flow.AddArcWithCapacityAndUnitCost(class_node(j), T, int(target_counts[j]), 0)

    # ============= (2) 设置供需 =============
    min_cost_flow.SetNodeSupply(S, bagsize)
    min_cost_flow.SetNodeSupply(T, -bagsize)

    # ============= (3) 求解 =============
    status = min_cost_flow.Solve()
    if status != min_cost_flow.OPTIMAL:
        raise RuntimeError("OR-Tools: 未找到最优解 (status = {})".format(status))

    # ============= (4) 根据结果构造 One-Hot 并计算 total_neglog =============
    best_onehot = torch.zeros((bagsize, n_classes), dtype=torch.int32)
    total_neglog = 0.0

    for arc_i in range(min_cost_flow.NumArcs()):
        flow = min_cost_flow.Flow(arc_i)
        if flow > 0:
            start = min_cost_flow.Tail(arc_i)
            end = min_cost_flow.Head(arc_i)

            # 样本节点到类别节点
            if 1 <= start <= bagsize and (bagsize + 1) <= end <= (bagsize + n_classes):
                i = start - 1
                j = end - (bagsize + 1)

                best_onehot[i, j] = 1

                # 取这条边的cost还原为浮点
                arc_unit_cost = min_cost_flow.UnitCost(arc_i)  # 整数
                real_cost = float(arc_unit_cost) / cost_scale  # 转回浮点
                total_neglog += real_cost

    return best_onehot, total_neglog


def find_k_solutions_and_aggregate(
        softmax_tensor: torch.Tensor,
        proportions: torch.Tensor,
        bagsize: int,
        k: int = 5,
        noise_scale: float = 0.05,
        cost_scale=10000,
        epsilon=1e-12,
        seed=None
):
    """
    1. 先求一次“最高置信度的解”（无扰动）
    2. 再重复 k-1 次加扰动，得到 k-1 个次优但仍较高置信度的解
    3. 利用方法1(平移)做数值稳定：按“总置信度” (exp of -neglog_i) 加权，融合这 k 个解成为一个伪标签分布
       其中一定包含原本最高置信度解

    返回：
    -------
    pseudo_label_soft: torch.Tensor
        [bagsize, n_classes] 的概率分布矩阵 (soft label)
    """

    # 转到 CPU 并转为 numpy
    softmax_cpu = softmax_tensor.detach().cpu().numpy()
    proportions_cpu = proportions.detach().cpu().numpy()
    n_classes = softmax_cpu.shape[1]

    # (1) 计算 target_counts
    target_counts = (proportions_cpu * bagsize).astype(int)
    remainder = bagsize - target_counts.sum()
    if remainder > 0:
        frac = proportions_cpu * bagsize - target_counts
        idx_sorted = np.argsort(frac)[::-1]  # 降序
        for idx in idx_sorted[:remainder]:
            target_counts[idx] += 1

    # (2) 第一次：无扰动 (noise_scale=0)
    best_onehot0, best_neglog0 = solve_mcf_once(
        softmax_cpu,
        target_counts,
        cost_scale=cost_scale,
        epsilon=epsilon,
        noise_scale=0.0,
        seed=seed
    )
    solutions = [(best_onehot0, best_neglog0)]

    # (3) 再找 k-1 个带扰动的解
    for rep in range(k - 1):
        this_seed = None if seed is None else (seed + rep + 1)
        onehot_i, neglog_i = solve_mcf_once(
            softmax_cpu,
            target_counts,
            cost_scale=cost_scale,
            epsilon=epsilon,
            noise_scale=noise_scale,
            seed=this_seed
        )
        solutions.append((onehot_i, neglog_i))

    # (4) 数值稳定地计算各解的权重
    # 收集所有 neglog
    all_neglogs = [item[1] for item in solutions]
    min_neglog = min(all_neglogs)

    # 对于每个解 i:
    #   weight_i = exp( -(neglog_i - min_neglog) )
    # = exp(min_neglog - neglog_i)
    # 然后再做归一化 -> 避免 exponent 正向溢出
    exp_vals = []
    for neglog_i in all_neglogs:
        exponent = -(neglog_i - min_neglog)  # = min_neglog - neglog_i
        exp_vals.append(math.exp(exponent))

    sum_exp = sum(exp_vals)
    weights = [v / sum_exp for v in exp_vals]  # 归一化权重

    # (5) 按权重对 onehot 做加权求和
    # 每个 onehot 的形状 [bagsize, n_classes]
    # 得到 soft label: [bagsize, n_classes]
    pseudo_label_soft = torch.zeros(
        (bagsize, n_classes), dtype=torch.float32
    )
    for w, (oh, _) in zip(weights, solutions):
        pseudo_label_soft += w * oh.float()

    # (6) 行方向归一化(可选)，使每行变成概率分布
    row_sum = pseudo_label_soft.sum(dim=1, keepdim=True) + 1e-12
    pseudo_label_soft = pseudo_label_soft / row_sum

    return pseudo_label_soft
def compute_single_bag_loss_dp_cuda(
    labels_p: torch.Tensor,    # (s, c)
    losses: torch.Tensor,      # (s, c)
    proportion: torch.Tensor   # (c,)
) -> torch.Tensor:
    """
    使用多维DP (在CUDA上) 精确计算:
      bag_loss = ( sum_{y \in Y}[ prod_j p_j(y_j ) * sum_j L_j(y_j ) ] )
                 / ( sum_{y \in Y}[ prod_j p_j(y_j) ] )
    其中 Y 表示所有满足各类别数目 = proportion[k]*s 的标签序列。

    参数:
    -------
    labels_p  : (s,c), 第 j 个样本在 c 个类别上的后验概率
    losses    : (s,c), 对应损失
    proportion: (c,),  每个类别应占比, sum_k proportion[k]=1

    返回:
    -------
    bag_loss: (标量Tensor), 该 bag 的期望损失
    """
    device = torch.device('cuda')  # 默认使用 GPU
    labels_p  = labels_p.to(device)
    losses    = losses.to(device)
    proportion= proportion.to(device)

    s, c = labels_p.shape

    # 计算每个类别所需的样本数 n_k
    class_counts = [int(round(proportion[k].item() * s)) for k in range(c)]
    if sum(class_counts) != s:
        raise ValueError(
            f"类别计数之和 != s, proportion * s 不为整数或round后有误差: {class_counts}"
        )

    # DP数组的形状: (s+1) x (n_0+1) x ... x (n_{c-1}+1)
    # DP[j, a1, a2, ..., ac] 表示“前 j 个样本的标签分配”满足类别 k 出现 a_k 次时，概率乘积之和
    # DP_loss 同理，但累加了 (概率乘积 * 损失和)

    shape = [s+1] + [cc+1 for cc in class_counts]
    DP       = torch.zeros(shape, dtype=torch.float, device=device)
    DP_loss  = torch.zeros(shape, dtype=torch.float, device=device)

    # 初始化 j=0 时，尚未分配任何样本，各类别的计数都是0
    init_idx = tuple([0] + [0]*c)
    DP[init_idx]      = 1.0
    DP_loss[init_idx] = 0.0

    # 主循环: j 从 1 到 s
    for j in range(1, s+1):
        # 第 j 个样本 (0-based 下标为 j-1)
        p_j = labels_p[j-1]  # shape=(c,)
        L_j = losses[j-1]    # shape=(c,)

        # 遍历上一层 j-1 的所有可行状态, 并尝试把第 j 个样本分配到类别 k
        # 这里用 itertools.product, 也可手写循环
        # idxs 格式: (j-1, n1, n2, ..., nc)
        for idxs in itertools.product(
            [j-1], *[range(cc+1) for cc in class_counts]
        ):
            prev_prob_sum = DP[idxs]
            if prev_prob_sum == 0.0:
                continue  # 无法到达的状态

            prev_loss_sum = DP_loss[idxs]
            n_vec = list(idxs[1:])  # (n1, n2, ..., nc)

            for k in range(c):
                # 如果要把第 j 个样本分配到类别 k, 那么该类别计数需+1
                if n_vec[k] < class_counts[k]:
                    new_n_vec = n_vec.copy()
                    new_n_vec[k] += 1

                    new_idx = tuple([j] + new_n_vec)

                    p_val = p_j[k]      # 该样本在类别 k 的后验概率
                    l_val = L_j[k]      # 该样本分配到类别 k 的损失

                    # 更新 DP[new_idx]
                    DP[new_idx]       += prev_prob_sum * p_val
                    DP_loss[new_idx]  += (prev_loss_sum * p_val
                                          + prev_prob_sum * p_val * l_val)

    # 取最终状态 (j=s, n1=class_counts[0],..., n_c=class_counts[c-1])
    final_idx = tuple([s] + class_counts)
    Z       = DP[final_idx]         # 分母: 概率乘积之和
    Z_loss  = DP_loss[final_idx]    # 分子: 概率乘积 * (损失和) 的累加

    if Z == 0.0:
        # 说明该标签配置总概率非常小(或=0), 返回0或其他处理
        return torch.tensor(0.0, device=device)
    else:
        return Z_loss / Z

# ============================
# 下面是一段简单测试代码示例
import torch

import torch


def random_swaps_and_softlabel_custom(
        onehot_labels: torch.Tensor,
        k: int,
        b: int,  # 新增超参数 b，表示要随机选择的索引对数量
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:
    """
    按照你的思路：
    1) 先把 onehot_labels 乘以 k，得到 k_onehot
    2) 进行 k 次循环，每次随机选择 2b 个索引，两两配对 (共 b 对)，执行：
        k_onehot[j] = k_onehot[j] - onehot[j] + onehot[i]
        k_onehot[i] = k_onehot[i] - onehot[j] + onehot[i]
    3) 对 k_onehot 做行归一化，得到 soft_labels

    参数
    ----
    onehot_labels : torch.Tensor
        形状 [bagsize, n_classes] 的 0/1 标签 (One-Hot 标签)
    k : int
        需要进行的随机互换次数
    b : int
        要随机选择的索引对数量 (即每次 2b 个索引，两两配对交换)
    device : torch.device
        设备 (CUDA or CPU)

    返回
    ----
    soft_labels : torch.Tensor
        形状 [bagsize, n_classes] 的软标签
    """

    # 将标签矩阵移动到指定设备
    onehot_labels = onehot_labels.to(device)
    bagsize, n_classes = onehot_labels.shape

    # 初始化 k_onehot = onehot_labels * k
    k_onehot = onehot_labels.clone() * k  # Shape: [bagsize, n_classes]

    if k > 0:
        for _ in range(k):
            # 随机选择 2b 个索引
            indices = torch.randperm(bagsize, device=device)[:2 * int(b)]  # 生成 2b 个互不相同的索引
            for idx in range(0, len(indices), 2):
                # 两两配对 (i, j)
                i = indices[idx]
                j = indices[idx + 1]

                # 执行交换操作
                k_onehot[j] = k_onehot[j] - onehot_labels[j] + onehot_labels[i]
                k_onehot[i] = k_onehot[i] - onehot_labels[j] + onehot_labels[i]

    # 行归一化，得到 soft_labels
    row_sum = k_onehot.sum(dim=1, keepdim=True).clamp_min(1e-12)
    soft_labels = k_onehot / row_sum

    return soft_labels

# ------------------- 测试示例 -------------------


def set_model(args):
    if args.dataset in ['CIFAR10', 'CIFAR100','miniImageNet','SVHN']:
        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
        if args.dataset in ['miniImageNet']:
            model = PaPiNet()
    else:
        model = LeNet5()
        model = WideResnet(
            n_classes=args.n_classes,
            k=args.wresnet_k,
            n=args.wresnet_n,
            proj=False
        )
    #model = ResNet18CIFAR10(num_classes=args.n_classes)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        # 兼容多种保存方式：优先 'model'，次选 'state_dict'，最后退回整包
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

        # 如有 DataParallel/DistributedDataParallel 的前缀 'module.'，统一去掉
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

        # 加载（允许跳过分类头）
        msg = model.load_state_dict(sd, strict=False)
        print('[load_state_dict] MISSING:', msg.missing_keys)
        print('[load_state_dict] UNEXPECTED:', msg.unexpected_keys)

        # 只允许分类头缺失，其它缺失或多余键则报错，方便尽快发现结构不匹配
        allowed_missing = {"classifier.weight", "classifier.bias"}
        unexpected_missing = set(msg.missing_keys) - allowed_missing
        if unexpected_missing or msg.unexpected_keys:
            raise ValueError(
                f'Unexpected missing keys: {unexpected_missing} | '
                f'unexpected keys: {msg.unexpected_keys}'
            )

        print(f'Loaded weights from checkpoint: {args.checkpoint}')

    model.cuda()
    model.train()

    if args.eval_ema:
        if args.dataset in ['CIFAR10', 'CIFAR100','SVHN']:
            ema_model = WideResnet(
                n_classes=args.n_classes,
                k=args.wresnet_k,
                n=args.wresnet_n,
                proj=False
            )
            if args.dataset in ['miniImageNet']:
                ema_model = models.resnet18(pretrained=False)
                ema_model.fc = nn.Linear(ema_model.fc.in_features, args.n_classes)

        else:
            ema_model = LeNet5()
            ema_model = WideResnet(
                n_classes=args.n_classes,
                k=args.wresnet_k,
                n=args.wresnet_n,
                proj=False
            )
        #ema_model = ResNet18CIFAR10(num_classes=args.n_classes)

        ema_model.cuda()
        ema_model.eval()
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()

    return model, criteria_x, criteria_u, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):
    """
    Momentum update of evaluation model (exponential moving average)
    """
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)


def llp_loss(labels_proportion, y):
    x = torch.tensor(labels_proportion, dtype=torch.float64).cuda()
    x = x.squeeze(0)  # 或者 x.squeeze()

    # Ensure y is also double

    y = y.double()
    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))
    mse_loss = torch.mean((x - y) ** 2)

    return cross_entropy


def custom_loss(probs, lambda_val=1.0):
    # probs is assumed to be a 2D tensor of shape (n, N_i)
    # where n is the number of rows and N_i is the number of columns

    # Compute the log of probs
    log_probs = torch.log(probs)

    # Multiply probs with log_probs element-wise
    product = -probs * log_probs

    # Compute the double sum
    loss = torch.sum(product)

    # Multiply by lambda
    loss = lambda_val * loss

    return loss


def thre_ema(thre, sum_values, ema):
    return thre * ema + (1 - ema) * sum_values


def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()  # 计算当前 mask 中的元素数量
    weight_decay = max(0, 1 - mask_count / max_mask_count)  # 线性衰减
    return initial_weight * weight_decay


def train_one_epoch(epoch,
                    bagsize,
                    n_classes,
                    model,
                    ema_model,
                    prob_list,
                    criteria_x,
                    criteria_u,
                    optim,
                    lr_schdlr,
                    dltrain_u,
                    args,
                    n_iters,
                    logger,
                    samp_ran
                    ):
    model.train()
    loss_u_meter = AverageMeter()
    loss_prop_meter = AverageMeter()
    thre_meter = AverageMeter()
    kl_meter = AverageMeter()
    kl_hard_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    # the number of correct pseudo-labels
    n_correct_u_lbs_meter = AverageMeter()
    # the number of confident unlabeled data
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()
    # the number of edges in the pseudo-label graph
    pos_meter = AverageMeter()
    samp_lb_meter, samp_p_meter = [], []
    for i in range(0, bagsize):
        x = CurrentValueMeter()
        y = CurrentValueMeter()
        samp_lb_meter.append(x)
        samp_p_meter.append(y)
    epoch_start = time.time()  # start time
    dl_u = iter(dltrain_u)
    n_iter = len(dltrain_u)

    for it in range(len(dltrain_u)):
        (var1, var2, var3, var4, var5) = next(dl_u)
        # var2 = torch.stack(var2)
        # print(var2)
        # print(f'var1:{var1.shape};\n var2: {var2.shape};\n var3: {var3.shape};\n var4: {var4.shape}')
        length = len(var2[0])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1, ims_u_strong01  = var1

        imsw, imss0, labels_real, labels_idx,indices_u = [], [], [], [],[]

        for i in range(length):
            imss0.append(ims_u_strong01[i])
            imsw.append(ims_u_weak1[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
        ims_u_strong0 = torch.cat(imss0, dim=0)
        lbs_u_real = torch.cat(labels_real, dim=0)
        label_proportions = [[] for _ in range(length)]
        lbs_u_real = lbs_u_real.cuda()
        lbs_idx = torch.cat(labels_idx, dim=0)
        lbs_idx = lbs_idx.cuda()

        positions = torch.nonzero(lbs_idx == 37821).squeeze()

        if positions.numel() != 0:
            head = positions - positions % bagsize
            rear = head + bagsize - 1

        for i in range(length):
            labels = []
            for j in range(n_classes):
                labels.append(var2[j][i])
            label_proportions[i].append(labels)

        # --------------------------------------
        btu = ims_u_weak.size(0)
        bt = 0
        #ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)

       # imgs = torch.cat([ims_u_weak, ims_u_strong0], dim=0).cuda()
        if args.dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
            ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)
            ims_u_strong0 = ims_u_strong0.permute(0, 2, 1, 3)
        imgs = torch.cat([ims_u_weak, ims_u_strong0], dim=0).cuda()

        out = model(imgs)
        if isinstance(out, (tuple, list)):
            logits = out[0]  # 取第一个作为 logits
        else:
            logits = out
        # logits_x = logits[:bt]
        logits_u_w, logits_u_s0 = torch.split(logits[0:], btu)

        # feats_x = features[:bt]
        #feats_u_w, feats_u_s0 = torch.split(features[0:], btu)

        # feats_x = fe
        # loss_x = criteria_x(logits_x, lbs_x)

        chunk_size = len(logits_u_w) // length
        batch_size = length
        # 分成 length 节
        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]

        # 打印分成的各节数据
        proportion = torch.empty((0, n_classes), dtype=torch.float64).cuda()
        batch_size = length

        # 循环生成 proportion 的每一行
        for i in range(length):
            pr = label_proportions[i][0]  # 获取每一行对应的列表
            pr = torch.stack(pr).cuda()  # 将列表转换为张量，并移动到 GPU 上
            proportion = torch.cat((proportion, pr.unsqueeze(0)))  # 按行拼接
        proportion = proportion.view(length, n_classes, 1)
        proportion = proportion.squeeze(-1)
        proportion = proportion.double()
        # 创建一个空的 PyTorch 向量用于保存 loss_p
        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        kl_divergence = torch.Tensor([]).cuda()
        kl_divergence = kl_divergence.double()
        kl_divergence_hard = torch.Tensor([]).cuda()
        kl_divergence_hard = kl_divergence_hard.double()
        # 假设您有一个名为 chunks 的列表，其中包含多个 chunk
        # 在循环中计算 loss_p 并添加到 all_loss_p 中
        onehot_flat_list = []

        for i, chunk in enumerate(chunks):
            labels_p = torch.softmax(chunk, dim=1)
            scores, lbs_u_guess = torch.max(labels_p, dim=1)
            opt_onehot = solve_optimal_onehot_with_proportions_torch(
                labels_p, proportion[i], bagsize, n_classes
            ).float().cuda()

            # —— 只新增这一行：把本轮 one-hot 展平收集 —— #
            onehot_flat_list.append(opt_onehot.reshape(-1))

            # 其余逻辑保持不变
            loss_p = cross_entropy_loss_torch(labels_p, opt_onehot)
            labels_p = torch.mean(labels_p, dim=0)
            loss_p = llp_loss(proportion[i], labels_p)

            label_prop = torch.tensor(label_proportions[i], dtype=torch.float64).cuda()
            loss_prop = torch.cat((loss_prop, loss_p.view(1)))

            label_prop += 1e-9
            labels_p += 1e-9
            log_labels_p = torch.log(labels_p)
            one_hot_matrix = F.one_hot(lbs_u_guess, num_classes=n_classes)
            one_hot_matrix = one_hot_matrix.float()
            one_hot_matrix = torch.mean(one_hot_matrix, dim=0)

            one_hot_matrix += 1e-9
            log_one_hot_matrix = torch.log(one_hot_matrix)

            kl_soft = F.kl_div(log_labels_p, label_prop, reduction='batchmean')
            kl_hard = F.kl_div(log_one_hot_matrix, label_prop, reduction='batchmean')
            kl_divergence = torch.cat((kl_divergence, kl_soft.view(1)))
            kl_divergence_hard = torch.cat((kl_divergence_hard, kl_hard.view(1)))

        # 循环后：得到所有轮的 one-hot 串成的一维向量
        onehot_1d = torch.cat(onehot_flat_list,
                              dim=0)  # shape: [num_bags * bagsize * n_classes]        kl_divergence = kl_divergence.mean()
        kl_divergence_hard = kl_divergence_hard.mean()
        loss_prop = loss_prop.mean()
        probs = torch.softmax(logits_u_w, dim=1)
        probs = probs.mean(dim=0)
        prior = torch.full_like(probs, 0.1).detach()
        prior = proportion.mean(dim=0).detach()

        loss_debais = llp_loss(prior, probs)
        x=loss_prop * bagsize
        loss = loss_prop
        B, C = logits_u_s0.size(0), logits_u_s0.size(1)
        pseudo_onehot = onehot_1d.view(-1, C).to(device=logits_u_w.device, dtype=logits_u_w.dtype)

        # 若 onehot_1d 是跨本轮所有 chunk 汇总的，确保长度匹配当前批：
        # assert pseudo_onehot.size(0) == B, "onehot_1d 与 logits_u_s0 的 batch 大小不匹配"

        # 用“软交叉熵”/KL 形式（对 one-hot 等价于标准 CE），不需要把 onehot 转成索引
        log_p = F.log_softmax(logits_u_s0, dim=1)  # [B, C]
        unsup_ce_per = -(pseudo_onehot * log_p).sum(dim=1)  # [B]

        # 如果你前面已经基于 logits_u_w 生成了置信度 mask（thr=0.95）：
        # max_probs, pseudo_idx = probs.max(dim=1)     # probs = softmax(logits_u_w, 1)
        probs = torch.softmax(logits_u_w, dim=1)

        scores, lbs_u_guess = torch.max(probs, dim=1)
        confidence = (probs * pseudo_onehot).sum(dim=1)  # [B]

        mask = (confidence >= args.thr).float()           # [B]
        # 那么就按掩码做平均，否则就直接平均
        loss_u = (criteria_u(logits_u_s0, pseudo_onehot) * mask).mean()

        loss = args.lam_u*loss_u + loss_prop
        with torch.no_grad():

            probs = torch.softmax(logits_u_w, dim=1)

            """
            max_probs, max_idx = torch.max(probs, dim=-1)
            # mask = max_probs.ge(p_cutoff * (class_acc[max_idx] + 1.) / 2).float()  # linear
            # mask = max_probs.ge(p_cutoff * (1 / (2. - class_acc[max_idx]))).float()  # low_limit
            mask = max_probs.ge(0.2+0.75 * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))).float()  # convex
            thre=0.2+0.75 * (classwise_acc[max_idx] / (2. - classwise_acc[max_idx]))

            thre_col = thre.view(-1, 1)  # 将 thre 变为列向量
            thre_row = thre.view(1, -1)  # 将 thre 变为行向量

            thre = torch.mm(thre_col, thre_row)
            delta = thre + (1 - thre) / (n_classes-1) * (1 - thre)
            thre=delta

            # mask = max_probs.ge(p_cutoff * (torch.log(class_acc[max_idx] + 1.) + 0.5)/(math.log(2) + 0.5)).float()  # concave
            select = max_probs.ge(args.thr).long()
            pseudo_lb=max_idx.long()
            pseudo_lb=pseudo_lb.cuda()
            if lbs_idx[select == 1].nelement() != 0:
                selected_label[lbs_idx[select == 1]] = pseudo_lb[select == 1]


            """
            # DA
            """
            prob_list.append(probs.mean(0))
            if len(prob_list)>32:
                prob_list.pop(0)
            prob_avg = torch.stack(prob_list,dim=0).mean(0)
            probs = probs / prob_avg
            probs = probs / probs.sum(dim=1, keepdim=True)
            """

            """
            probs是分类器对弱增强输出对 softmax结果
            """

            """
            probs_orig = probs.clone()

            if epoch>0 or it>args.queue_batch: # memory-smoothing
                A = torch.exp(torch.mm(feats_u_w, queue_feats.t())/args.temperature)
                A = A/A.sum(1,keepdim=True)
                probs = args.alpha*probs + (1-args.alpha)*torch.mm(A, queue_probs)
            """

            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()
            if positions.numel() != 0:
                for i in range(0, bagsize):
                    samp_lb_meter[i].update(lbs_u_guess[head + i].item())
                    samp_p_meter[i].update(scores[head + i].item())
            """
            feats_w=feats_u_w
            probs_w=probs_orig

            # update memory bank
            n = bt+btu
            queue_feats[queue_ptr:queue_ptr + n,:] = feats_w
            queue_probs[queue_ptr:queue_ptr + n,:] = probs_w
            queue_ptr = (queue_ptr+n)%args.queue_size
            """

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_prop_meter.update(loss.item())
        mask_meter.update(mask.mean().item())
        kl_meter.update(kl_divergence.mean().item())
        kl_hard_meter.update(kl_divergence_hard.mean().item())
        lbs_u_guess = pseudo_onehot.argmax(dim=1)  # [N] LongTensor

        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        total = mask.sum()
        corr_u_lb = corr_u_lb /total
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss: {:.3f}. kl: {:.3f}. kl_hard:{:.3f}.  acc:{:.3f}  "
                        "LR: {:.3f}. Time: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_prop_meter.avg, kl_meter.avg,
                n_correct_u_lbs_meter.avg,
                kl_hard_meter.avg, lr_log, t))

            epoch_start = time.time()

    return loss_prop_meter.avg, n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, kl_meter.avg, kl_hard_meter.avg


from sklearn.metrics import f1_score
import torch


def evaluate(model, ema_model, dataloader,dataset):
    model.eval()

    top1_meter = AverageMeter()
    ema_top1_meter = AverageMeter()
    top5_meter = AverageMeter()
    ema_top5_meter = AverageMeter()
    loss_meter = AverageMeter()  # 假设你有一个 AverageMeter 类来计算均值

    all_preds = []  # 用于存储所有预测的类别
    all_labels = []  # 用于存储所有真实的类别
    ema_all_preds = []  # 用于存储 EMA 模型的所有预测类别
    ema_all_labels = []  # 用于存储 EMA 模型的所有真实类别

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            #ims = ims.permute(0, 2, 1, 3)
            if dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
                ims = ims.permute(0, 2, 1, 3)

            # 计算模型的输出和损失
            out = model(ims)
            if isinstance(out, (tuple, list)):
                logits = out[0]  # 取第一个作为 logits
            else:
                logits = out
            loss = torch.nn.CrossEntropyLoss()(logits, lbs)

            # 更新交叉熵损失的累加器
            loss_meter.update(loss.item())

            # 获取模型预测结果并更新 top1 和 top5
            scores = torch.softmax(logits, dim=1)
            preds = scores.argmax(dim=1)  # 获取预测类别
            all_preds.extend(preds.cpu().numpy())  # 保存预测结果
            all_labels.extend(lbs.cpu().numpy())  # 保存真实标签

            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            # 计算 EMA 模型的结果
            if ema_model is not None:
                out = ema_model(ims)
                if isinstance(out, (tuple, list)):
                    ema_logits = out[0]  # 取第一个作为 logits
                else:
                    ema_logits = out

                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_preds = ema_scores.argmax(dim=1)  # 获取 EMA 模型预测类别
                ema_all_preds.extend(ema_preds.cpu().numpy())  # 保存 EMA 模型的预测结果
                ema_all_labels.extend(lbs.cpu().numpy())  # 保存真实标签

                ema_top1, ema_top5 = accuracy(ema_scores, lbs, (1, 5))
                ema_top1_meter.update(ema_top1.item())

    # 计算 top1 的 macro F1 分数
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    ema_macro_f1 = f1_score(ema_all_labels, ema_all_preds, average='macro') if ema_model is not None else None

    return top1_meter.avg,  top5_meter.avg,  loss_meter.avg, macro_f1

def main():
    parser = argparse.ArgumentParser(description='LLP_DC Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default="CIFAR100",
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=100                                                                                                             ,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='1')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize', type=int, default=16,
                        help='train bag size of samples')
    parser.add_argument('--eval-ema', default=False, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)


    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=10,
                        help='seed for random behaviors, no seed if negtive')

    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')
    parser.add_argument('--lam-u', type=float, default=0.5,
                        help='coefficient of proportion loss')

    parser.add_argument('--thr', type=float, default=0.6,
                        help='pseudo label threshold')

    parser.add_argument('--exp-dir', default='LLP_DC', type=str, help='experiment id')
    parser.add_argument('--checkpoint', default='', type=str, help='use pretrained model')
    parser.add_argument('--folds', default='2', type=str, help='number of dataset')
    args = parser.parse_args()

    logger, output_dir = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    tb_logger = tensorboard_logger.Logger(logdir=output_dir, flush_secs=2)
    samp_ran = 37821
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    n_iters_per_epoch = args.n_imgs_per_epoch  # 1024

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")

    model, criteria_x, criteria_u, ema_model = set_model(args)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_u, dataset_length,_ = get_train_loader(args.n_classes,
                                                 args.dataset, args.batchsize, args.bagsize, root=args.root,
                                                 method='L^2P-AHIL',
                                                 supervised=False)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root)
    n_iters_all = len(dltrain_u) * args.n_epoches
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)
    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    optim = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)

    lr_schdlr = WarmupCosineLrScheduler(optim, n_iters_all, warmup_iter=0)

    # memory bank
    args.queue_size = 5120
    queue_feats = torch.zeros(args.queue_size, args.low_dim).cuda()
    queue_probs = torch.zeros(args.queue_size, args.n_classes).cuda()
    queue_ptr = 0

    # for distribution alignment
    prob_list = []

    train_args = dict(
        model=model,
        ema_model=ema_model,
        prob_list=prob_list,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        optim=optim,
        lr_schdlr=lr_schdlr,
        dltrain_u=dltrain_u,
        args=args,
        n_iters=n_iters_per_epoch,
        logger=logger
    )

    best_acc = -1
    best_acc_5 = -1
    best_epoch_5 = 0

    best_epoch = 0

    logger.info('-----------start training--------------')
    for epoch in range(args.n_epoches):
        loss_prob, n_correct_u_lbs, n_strong_aug, mask_mean, num_pos, samp_lb = \
            train_one_epoch(epoch, bagsize=args.bagsize, n_classes=args.n_classes, **train_args, samp_ran=samp_ran
                            )

        top1, top5, loss_test,macro = evaluate(model, ema_model, dlval,args.dataset)
        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)

        tb_logger.log_value('loss_test', loss_test, epoch)
        tb_logger.log_value('macro', macro, epoch)


        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if best_acc_5 < top5:
            best_acc_5 = top5
            best_epoch_5 = epoch
        logger.info(
            "Epoch {}.loss_test: {:.4f}. Acc: {:.4f}.  Macro_F1: {:.4f}. best_acc: {:.4f} in epoch{},Acc_5: {:.4f}.  best_acc_5: {:.4f} in epoch{},".
            format(epoch, loss_test, top1,macro, best_acc, best_epoch, top5, best_acc_5, best_epoch_5))

        if epoch % 123989 == 0:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'checkpoint_%02d.pth' % epoch))


if __name__ == '__main__':
    main()