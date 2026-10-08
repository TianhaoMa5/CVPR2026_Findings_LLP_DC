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
    __slots__ = ('to', 'rev', 'capacity', 'cost', 'flow')

    def __init__(self, to, rev, capacity, cost):
        self.to = to
        self.rev = rev
        self.capacity = capacity
        self.cost = cost
        self.flow = 0


class MinCostMaxFlow:

    def __init__(self, n):
        self.n = n
        self.adjacency = [[] for _ in range(n)]

    def add_edge(self, u, v, capacity, cost):
        self.adjacency[u].append(Edge(v, len(self.adjacency[v]), capacity, cost))
        self.adjacency[v].append(Edge(u, len(self.adjacency[u]) - 1, 0, -cost))

    def min_cost_max_flow(self, source, sink):
        flow, cost = 0, 0
        INF = 10 ** 14

        while True:
            dist = [math.inf] * self.n
            in_queue = [False] * self.n
            parent_node = [-1] * self.n
            parent_edge = [-1] * self.n

            dist[source] = 0
            queue = [source]
            in_queue[source] = True

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
                break

            augment = INF
            v = sink
            while v != source:
                e_idx = parent_edge[v]
                u = parent_node[v]
                e = self.adjacency[u][e_idx]
                augment = min(augment, e.capacity - e.flow)
                v = u

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
    log_softmax = torch.log(softmax_matrix + 1e-12)

    cross_entropy = -torch.sum(onehot_labels * log_softmax, dim=1)

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
    assert isinstance(softmax_tensor, torch.Tensor), "softmax_tensor must be a PyTorch tensor"
    assert isinstance(proportions, torch.Tensor), "proportions must be a PyTorch tensor"

    softmax_cpu = softmax_tensor.detach().cpu().numpy()
    proportions_cpu = proportions.detach().cpu().numpy()

    if not math.isclose(proportions_cpu.sum(), 1.0, rel_tol=1e-6, abs_tol=1e-9):
        raise ValueError("The sum of proportions must be close to 1.")

    target_counts = (proportions_cpu * bagsize).astype(int)
    remaining = bagsize - target_counts.sum()

    if remaining > 0:
        fractional_parts = proportions_cpu * bagsize - target_counts
        sorted_indices = fractional_parts.argsort()[::-1]
        for idx in sorted_indices[:remaining]:
            target_counts[idx] += 1

    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    S = 0
    T = bagsize + n_classes + 1

    def sample_node(i):
        return i + 1

    def class_node(j):
        return bagsize + 1 + j

    for i in range(bagsize):
        min_cost_flow.AddArcWithCapacityAndUnitCost(S, sample_node(i), 1, 0)

    for i in range(bagsize):
        for j in range(n_classes):
            p_ij = max(softmax_cpu[i, j], epsilon)
            cost = int(-math.log(p_ij) * cost_scale)
            min_cost_flow.AddArcWithCapacityAndUnitCost(sample_node(i), class_node(j), 1, cost)

    for j in range(n_classes):
        min_cost_flow.AddArcWithCapacityAndUnitCost(
            class_node(j), T, int(target_counts[j]), 0
        )
    min_cost_flow.SetNodeSupply(S, bagsize)
    min_cost_flow.SetNodeSupply(T, -bagsize)

    status = min_cost_flow.Solve()
    if status != min_cost_flow.OPTIMAL:
        raise RuntimeError("OR-Tools: no optimal solution found")

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
    if seed is not None:
        random.seed(seed)

    bagsize, n_classes = softmax_cpu.shape
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # S = 0, T = bagsize + n_classes + 1
    S = 0
    T = bagsize + n_classes + 1

    def sample_node(i):
        return 1 + i

    def class_node(j):
        return bagsize + 1 + j

    for i in range(bagsize):
        min_cost_flow.AddArcWithCapacityAndUnitCost(S, sample_node(i), 1, 0)

    for i in range(bagsize):
        for j in range(n_classes):
            p_ij = max(softmax_cpu[i, j], epsilon)
            base_cost = -math.log(p_ij)

            if noise_scale > 0:
                noise = random.gauss(0, noise_scale)
                final_cost = base_cost + noise
            else:
                final_cost = base_cost

            arc_cost = int(final_cost * cost_scale)
            min_cost_flow.AddArcWithCapacityAndUnitCost(sample_node(i), class_node(j), 1, arc_cost)

    for j in range(n_classes):
        min_cost_flow.AddArcWithCapacityAndUnitCost(class_node(j), T, int(target_counts[j]), 0)

    min_cost_flow.SetNodeSupply(S, bagsize)
    min_cost_flow.SetNodeSupply(T, -bagsize)

    status = min_cost_flow.Solve()
    if status != min_cost_flow.OPTIMAL:
        raise RuntimeError("OR-Tools: no optimal solution found (status = {})".format(status))

    best_onehot = torch.zeros((bagsize, n_classes), dtype=torch.int32)
    total_neglog = 0.0

    for arc_i in range(min_cost_flow.NumArcs()):
        flow = min_cost_flow.Flow(arc_i)
        if flow > 0:
            start = min_cost_flow.Tail(arc_i)
            end = min_cost_flow.Head(arc_i)

            if 1 <= start <= bagsize and (bagsize + 1) <= end <= (bagsize + n_classes):
                i = start - 1
                j = end - (bagsize + 1)

                best_onehot[i, j] = 1

                arc_unit_cost = min_cost_flow.UnitCost(arc_i)
                real_cost = float(arc_unit_cost) / cost_scale
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

    softmax_cpu = softmax_tensor.detach().cpu().numpy()
    proportions_cpu = proportions.detach().cpu().numpy()
    n_classes = softmax_cpu.shape[1]

    target_counts = (proportions_cpu * bagsize).astype(int)
    remainder = bagsize - target_counts.sum()
    if remainder > 0:
        frac = proportions_cpu * bagsize - target_counts
        idx_sorted = np.argsort(frac)[::-1]
        for idx in idx_sorted[:remainder]:
            target_counts[idx] += 1

    best_onehot0, best_neglog0 = solve_mcf_once(
        softmax_cpu,
        target_counts,
        cost_scale=cost_scale,
        epsilon=epsilon,
        noise_scale=0.0,
        seed=seed
    )
    solutions = [(best_onehot0, best_neglog0)]

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

    all_neglogs = [item[1] for item in solutions]
    min_neglog = min(all_neglogs)

    #   weight_i = exp( -(neglog_i - min_neglog) )
    # = exp(min_neglog - neglog_i)
    exp_vals = []
    for neglog_i in all_neglogs:
        exponent = -(neglog_i - min_neglog)  # = min_neglog - neglog_i
        exp_vals.append(math.exp(exponent))

    sum_exp = sum(exp_vals)
    weights = [v / sum_exp for v in exp_vals]

    pseudo_label_soft = torch.zeros(
        (bagsize, n_classes), dtype=torch.float32
    )
    for w, (oh, _) in zip(weights, solutions):
        pseudo_label_soft += w * oh.float()

    row_sum = pseudo_label_soft.sum(dim=1, keepdim=True) + 1e-12
    pseudo_label_soft = pseudo_label_soft / row_sum

    return pseudo_label_soft
def compute_single_bag_loss_dp_cuda(
    labels_p: torch.Tensor,    # (s, c)
    losses: torch.Tensor,      # (s, c)
    proportion: torch.Tensor   # (c,)
) -> torch.Tensor:
    device = torch.device('cuda')
    labels_p  = labels_p.to(device)
    losses    = losses.to(device)
    proportion= proportion.to(device)

    s, c = labels_p.shape

    class_counts = [int(round(proportion[k].item() * s)) for k in range(c)]
    if sum(class_counts) != s:
        raise ValueError(
            f"The sum of class counts does not equal s; proportion * s is non-integer or differs after rounding: {class_counts}"
        )


    shape = [s+1] + [cc+1 for cc in class_counts]
    DP       = torch.zeros(shape, dtype=torch.float, device=device)
    DP_loss  = torch.zeros(shape, dtype=torch.float, device=device)

    init_idx = tuple([0] + [0]*c)
    DP[init_idx]      = 1.0
    DP_loss[init_idx] = 0.0

    for j in range(1, s+1):
        p_j = labels_p[j-1]  # shape=(c,)
        L_j = losses[j-1]    # shape=(c,)

        for idxs in itertools.product(
            [j-1], *[range(cc+1) for cc in class_counts]
        ):
            prev_prob_sum = DP[idxs]
            if prev_prob_sum == 0.0:
                continue

            prev_loss_sum = DP_loss[idxs]
            n_vec = list(idxs[1:])  # (n1, n2, ..., nc)

            for k in range(c):
                if n_vec[k] < class_counts[k]:
                    new_n_vec = n_vec.copy()
                    new_n_vec[k] += 1

                    new_idx = tuple([j] + new_n_vec)

                    p_val = p_j[k]
                    l_val = L_j[k]

                    DP[new_idx]       += prev_prob_sum * p_val
                    DP_loss[new_idx]  += (prev_loss_sum * p_val
                                          + prev_prob_sum * p_val * l_val)

    final_idx = tuple([s] + class_counts)
    Z       = DP[final_idx]
    Z_loss  = DP_loss[final_idx]

    if Z == 0.0:
        return torch.tensor(0.0, device=device)
    else:
        return Z_loss / Z

# ============================
import torch

import torch


def random_swaps_and_softlabel_custom(
        onehot_labels: torch.Tensor,
        k: int,
        b: int,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> torch.Tensor:

    onehot_labels = onehot_labels.to(device)
    bagsize, n_classes = onehot_labels.shape

    k_onehot = onehot_labels.clone() * k  # Shape: [bagsize, n_classes]

    if k > 0:
        for _ in range(k):
            indices = torch.randperm(bagsize, device=device)[:2 * int(b)]
            for idx in range(0, len(indices), 2):
                i = indices[idx]
                j = indices[idx + 1]

                k_onehot[j] = k_onehot[j] - onehot_labels[j] + onehot_labels[i]
                k_onehot[i] = k_onehot[i] - onehot_labels[j] + onehot_labels[i]

    row_sum = k_onehot.sum(dim=1, keepdim=True).clamp_min(1e-12)
    soft_labels = k_onehot / row_sum

    return soft_labels



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
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))

        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

        msg = model.load_state_dict(sd, strict=False)
        print('[load_state_dict] MISSING:', msg.missing_keys)
        print('[load_state_dict] UNEXPECTED:', msg.unexpected_keys)

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
    x = x.squeeze(0)

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
    mask_count = mask.sum().item()
    weight_decay = max(0, 1 - mask_count / max_mask_count)
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
            logits = out[0]
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
        chunks = [logits_u_w[i * chunk_size:(i + 1) * chunk_size] for i in range(length)]

        proportion = torch.empty((0, n_classes), dtype=torch.float64).cuda()
        batch_size = length

        for i in range(length):
            pr = label_proportions[i][0]
            pr = torch.stack(pr).cuda()
            proportion = torch.cat((proportion, pr.unsqueeze(0)))
        proportion = proportion.view(length, n_classes, 1)
        proportion = proportion.squeeze(-1)
        proportion = proportion.double()
        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        kl_divergence = torch.Tensor([]).cuda()
        kl_divergence = kl_divergence.double()
        kl_divergence_hard = torch.Tensor([]).cuda()
        kl_divergence_hard = kl_divergence_hard.double()
        onehot_flat_list = []

        for i, chunk in enumerate(chunks):
            labels_p = torch.softmax(chunk, dim=1)
            scores, lbs_u_guess = torch.max(labels_p, dim=1)
            opt_onehot = solve_optimal_onehot_with_proportions_torch(
                labels_p, proportion[i], bagsize, n_classes
            ).float().cuda()

            onehot_flat_list.append(opt_onehot.reshape(-1))

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


        log_p = F.log_softmax(logits_u_s0, dim=1)  # [B, C]
        unsup_ce_per = -(pseudo_onehot * log_p).sum(dim=1)  # [B]

        # max_probs, pseudo_idx = probs.max(dim=1)     # probs = softmax(logits_u_w, 1)
        probs = torch.softmax(logits_u_w, dim=1)

        scores, lbs_u_guess = torch.max(probs, dim=1)
        confidence = (probs * pseudo_onehot).sum(dim=1)  # [B]

        mask = (confidence >= args.thr).float()           # [B]
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

            thre_col = thre.view(-1, 1)
            thre_row = thre.view(1, -1)

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
    loss_meter = AverageMeter()

    all_preds = []
    all_labels = []
    ema_all_preds = []
    ema_all_labels = []

    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            #ims = ims.permute(0, 2, 1, 3)
            if dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
                ims = ims.permute(0, 2, 1, 3)

            out = model(ims)
            if isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            loss = torch.nn.CrossEntropyLoss()(logits, lbs)

            loss_meter.update(loss.item())

            scores = torch.softmax(logits, dim=1)
            preds = scores.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

            top1, top5 = accuracy(scores, lbs, (1, 5))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            if ema_model is not None:
                out = ema_model(ims)
                if isinstance(out, (tuple, list)):
                    ema_logits = out[0]
                else:
                    ema_logits = out

                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_preds = ema_scores.argmax(dim=1)
                ema_all_preds.extend(ema_preds.cpu().numpy())
                ema_all_labels.extend(lbs.cpu().numpy())

                ema_top1, ema_top5 = accuracy(ema_scores, lbs, (1, 5))
                ema_top1_meter.update(ema_top1.item())

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