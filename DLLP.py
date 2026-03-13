from __future__ import print_function
import random

import time
import argparse
import os
import sys
import os, csv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, OrderedDict
from WideResNet import WideResnet
from datasets.cifar import get_train_loader, get_val_loader
from utils import accuracy, setup_default_logging, AverageMeter, CurrentValueMeter, WarmupCosineLrScheduler
import tensorboard_logger
import torch.multiprocessing as mp
from LeNet import LeNet5,MLPDropIn
from torchvision import models

import torch
import math
from model import CNNBackbone
from papi import PaPiNet



import copy


def cross_entropy_loss_torch(softmax_matrix, onehot_labels):

    log_softmax = torch.log(softmax_matrix + 1e-12)

    cross_entropy = -torch.sum(onehot_labels * log_softmax, dim=1)

    mean_loss = torch.mean(cross_entropy)
    return mean_loss
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)



def set_model(args,input_dim):
    if args.dataset in ['CIFAR10', 'SVHN', 'CIFAR100', 'miniImageNet']:
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, args.n_classes)
        model = WideResnet(n_classes=args.n_classes, k=args.wresnet_k, n=args.wresnet_n, proj=True)
        if args.dataset in ['miniImageNet']:
            model = PaPiNet()

    elif args.dataset in [
        'Corel16k', 'Corel5k', 'Delicious', 'Bookmarks',
        'Eurlex_DC', 'Eurlex_SM', 'Scene', 'Yeast'
    ]:
        model = LinearClassifier(
            input_dim=input_dim,
            output_dim=args.n_classes
        )

    else:
        model = LeNet5()


    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        msg = model.load_state_dict(ckpt, strict=False)
        assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
        print('loaded from checkpoint:', args.checkpoint)

    model.train().cuda()

    if args.eval_ema:
        ema_model = copy.deepcopy(model).cuda().eval()
        for p in ema_model.parameters():
            p.requires_grad = False
    else:
        ema_model = None

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()

    return model, criteria_x, criteria_u, ema_model


@torch.no_grad()
def ema_model_update(model, ema_model, ema_m):

    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval * ema_m + param_train.detach() * (1 - ema_m))

    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)


def llp_loss(labels_proportion, y):
    x = torch.tensor(labels_proportion, dtype=torch.float64).cuda()
    x = x.squeeze(0)



    y = y.double()
    cross_entropy = torch.sum(-x * (torch.log(y) + 1e-7))
    mse_loss = torch.mean((x - y) ** 2)

    return cross_entropy


def custom_loss(probs, lambda_val=1.0):

    log_probs = torch.log(probs)

    product = -probs * log_probs

    loss = torch.sum(product)

    loss = lambda_val * loss

    return loss



def thre_ema(thre, sum_values, ema):
    return thre * ema + (1 - ema) * sum_values


def weight_decay_with_mask(mask, initial_weight, max_mask_count):
    mask_count = mask.sum().item()
    weight_decay = max(0, 1 - mask_count / max_mask_count)
    return initial_weight * weight_decay
import torch

def llp_second_order_ce_mass_shape(labels_p_batch, proportion, bagsize, use_counts=False, eps=1e-8):
    """
    labels_p_batch: [B, s, C]  — 每袋 s 个实例的 softmax 概率
    proportion    : [B, C]     — 袋比例 r_c；若 use_counts=True，则 proportion*s 视为计数 K_c
    bagsize (s)   : int
    返回: loss_2nd_total, loss_mass, loss_shape
    """
    B, s, C = labels_p_batch.shape
    s = int(s)
    # ---------- 预测端：同类对概率 t_hat_c ----------
    S1 = labels_p_batch.sum(dim=1)                  # [B,C],  \sum_i p_ic
    S2 = (labels_p_batch**2).sum(dim=1)             # [B,C],  \sum_i p_ic^2
    denom = (s * (s - 1.0)) + eps
    t_hat = (S1**2 - S2) / denom                    # [B,C], 同类对概率
    t_hat = t_hat.clamp(eps, 1 - eps)

    # ---------- 目标端：t_c ----------
    if use_counts:
        # proportion 是 K/s
        K = (proportion * s).to(labels_p_batch.dtype)       # [B,C]
        t = (K * (K - 1.0)) / denom                         # [B,C]
    else:
        r = (proportion / (proportion.sum(dim=1, keepdim=True) + eps)).to(labels_p_batch.dtype)
        t = (r ** 2)                                        # [B,C] 推荐用 r^2，始终 ∈[0,1]
    t = t.clamp(eps, 1 - eps)

    # ---------- mass：同类对总质量 ----------
    q2_hat = t_hat.sum(dim=1, keepdim=True)                 # [B,1]
    q2     = t.sum(dim=1, keepdim=True)                     # [B,1]
    loss_mass = - ( q2 * (q2_hat + eps).log() + (1 - q2) * (1 - q2_hat + eps).log() ).mean()

    # ---------- shape：在“同类对”条件下的类分布 ----------
    pi_hat = (t_hat / (q2_hat + eps)).clamp(eps, 1 - eps)   # [B,C]
    pi     = (t     / (q2     + eps)).clamp(eps, 1 - eps)   # [B,C]
    loss_shape = - (pi * (pi_hat + eps).log()).sum(dim=1).mean()

    loss_2nd = loss_mass + loss_shape
    return loss_2nd, loss_mass, loss_shape
def llp_loss_batch(labels_proportion, y, reduce: str = "mean", eps: float = 1e-7):

    x = labels_proportion.to(dtype=torch.float64, device=y.device)
    y = y.to(dtype=torch.float64)

    cross_entropy = - (x * (torch.log(y.clamp_min(eps)))).sum(dim=1)  # [B]

    mse_loss = ((x - y) ** 2).mean(dim=1)  # [B]


    loss = cross_entropy
    if reduce is None:
        return loss
    elif reduce == "mean":
        return loss.mean()
    elif reduce == "sum":
        return loss.sum()
    else:
        raise ValueError("reduce must be None|'mean'|'sum'")

import torch

def llp_second_order_only_ce(
    labels_p_batch: torch.Tensor,   # [B, s, C]  每袋 s 个样本的 softmax 概率
    proportion: torch.Tensor,       # [B, C]     袋比例 r_c；若 use_counts=True，则 proportion*s≈K_c
    bagsize: int,                   # s
    mode: str = "massshape",        # "massshape" | "matrix_ce"
    use_counts: bool = False,       # 只有比例时 False；有整数计数时 True
    eps: float = 1e-8,
):
    """
    返回: loss_2nd, dict(可选日志)
    只包含二阶项，不含一阶 bag-CE。
    """
    B, s, C = labels_p_batch.shape
    P = labels_p_batch
    r = proportion.to(P.dtype)
    r = r / (r.sum(1, keepdim=True) + eps)     # 防御性归一化

    # 公共量
    S1 = P.sum(1)                               # [B,C]
    S2 = (P**2).sum(1)                          # [B,C]
    denom = s * (s - 1.0) + eps

    if mode.lower() == "massshape":
        # per-class 同类对概率（预测）
        t_hat = (S1**2 - S2) / denom            # [B,C]
        t_hat = t_hat.clamp(eps, 1 - eps)

        # 目标：比例更稳（始终∈[0,1]）；若有整数计数则改用 use_counts=True
        if use_counts:
            K = (proportion * s).to(P.dtype)        # [B,C]
            t = (K * (K - 1.0)) / denom             # [B,C]
        else:
            t = (r**2)                               # [B,C]
        t = t.clamp(eps, 1 - eps)

        # mass：同类对总质量（同类 vs 异类，二元 CE）
        q2_hat = t_hat.sum(1, keepdim=True)          # [B,1]
        q2     = t.sum(1, keepdim=True)              # [B,1]
        loss_mass  = - ( q2 * (q2_hat+eps).log() + (1-q2) * (1-q2_hat+eps).log() ).mean()

        # shape：在“同类对”条件下的类分布（多类 CE）
        pi_hat = (t_hat / (q2_hat + eps)).clamp(eps, 1 - eps)   # [B,C]
        pi     = (t     / (q2     + eps)).clamp(eps, 1 - eps)   # [B,C]
        loss_shape = - (pi * (pi_hat + eps).log()).sum(1).mean()

        loss_2nd = loss_mass + loss_shape
        logs = {"mass": loss_mass.detach(), "shape": loss_shape.detach(),
                "q2_hat": q2_hat.mean().detach(), "q2": q2.mean().detach()}

    elif mode.lower() == "matrix_ce":
        # 整体 CxC 有序对矩阵（含异类对）
        S1S1T = torch.einsum('bc,bd->bcd', S1, S1)          # [B,C,C]
        S2mat = torch.einsum('bmc,bmd->bcd', P, P)          # \sum_i p_i p_i^T
        Pi_hat = (S1S1T - S2mat) / denom                    # [B,C,C]
        Pi_hat = Pi_hat.clamp(eps, 1 - eps)

        if use_counts:
            K = (proportion * s).to(P.dtype)                # [B,C]
            KKt = torch.einsum('bc,bd->bcd', K, K)          # [B,C,C]
            Pi = (KKt - torch.diag_embed(K)) / denom        # [B,C,C]
        else:
            # 期望目标（无整数计数时）：E[(KK^T - diag K)] / s(s-1) ≈ r r^T - diag(r)/s
            rrT = torch.einsum('bc,bd->bcd', r, r)
            Pi = rrT - torch.diag_embed(r / max(s, 1.0))
        Pi = Pi.clamp(eps, 1 - eps)

        # 条目级 CE（矩阵 KL 等价 up to 常数）
        loss_2nd = - (Pi * (Pi_hat+eps).log() + (1-Pi) * (1-Pi_hat+eps).log()).sum((1,2)).mean()
        logs = {}

    else:
        raise ValueError("mode must be 'massshape' or 'matrix_ce'")

    return loss_2nd, logs

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
    entropy_meter = AverageMeter()
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
        var1 = var1[0]
        # var2 = torch.stack(var2)
        # print(var2)
        # print(f'var1:{var1.shape};\n var2: {var2.shape};\n var3: {var3.shape};\n var4: {var4.shape}')
        length = len(var2[0])

        """
        pseudo_counter = Counter(selected_label.tolist())
        for i in range(args.n_classes):
            classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

        """
        ims_u_weak1 = var1

        imsw, labels_real, labels_idx = [], [], []  # $$$$$$$$$$$$$

        for i in range(length):
            imsw.append(ims_u_weak1[i])
            labels_real.append(var3[i])
            labels_idx.append(var4[i])
        ims_u_weak = torch.cat(imsw, dim=0)
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

        if args.dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
            ims_u_weak = ims_u_weak.permute(0, 2, 1, 3)
        bt = 0
        imgs = torch.cat([ims_u_weak], dim=0).cuda()
        logits,_ = model(imgs)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]  # 取真正的分类输出那个 tensor

        # logits_x = logits[:bt]
        logits_u_w = torch.split(logits[0:], btu)
        logits_u_w = logits_u_w[0]
        proportion = torch.stack([torch.stack(lp[0]) for lp in label_proportions]).cuda()
        proportion = proportion.view(length, n_classes, 1)
        proportion = proportion.squeeze(-1)
        proportion = proportion.double()

        probs = torch.softmax(logits_u_w, dim=-1)  # 或 dim=1
        # loss_x = criteria_x(logits_x, lbs_x)
        N, C = probs.shape
        s = bagsize
        assert N % s == 0, f"N={N} 不能被 bagsize={s} 整除"
        B = N // s

        # 变成 [B, s, C]
        labels_p_batch = probs.contiguous().view(B, s, C)
        bag_preds = labels_p_batch.mean(dim=1)  # [B, C]

        # loss_x = criteria_x(logits_x, lbs_x)

        chunk_size = len(logits_u_w) // length

        loss_prop = torch.Tensor([]).cuda()
        loss_prop = loss_prop.double()
        kl_divergence = torch.Tensor([]).cuda()
        kl_divergence = kl_divergence.double()
        kl_divergence_hard = torch.Tensor([]).cuda()
        kl_divergence_hard = kl_divergence_hard.double()


        loss_2, logs = llp_second_order_only_ce(
            labels_p_batch=labels_p_batch,
            proportion=proportion,
            bagsize=bagsize,
            mode="massshape",  # 或 "matrix_ce"
            use_counts=False,  # 只有比例时 False；有整数计数再切 True
            eps=1e-8,
        )
        loss_1 = llp_loss_batch(proportion, bag_preds, reduce="mean")
        loss = loss_1#+loss_2
        x=1.2
        kl_divergence = x
        kl_divergence_hard = x
        loss_prop = loss_prop.mean()
        probs = torch.softmax(logits_u_w, dim=1)
        probs = probs.mean(dim=0)
        prior = torch.full_like(probs, 0.1).detach()
        prior = proportion.mean(dim=0).detach()
        entropy = -(probs * probs.log()).sum(dim=-1)  # shape: [batch_size]

        # (Optional) average entropy over the batch
        mean_entropy = entropy.mean()
        with torch.no_grad():

            probs = torch.softmax(logits_u_w, dim=1)

            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(args.thr).float()

        optim.zero_grad()
        loss.backward()
        optim.step()
        lr_schdlr.step()

        if args.eval_ema:
            with torch.no_grad():
                ema_model_update(model, ema_model, args.ema_m)
        loss_prop_meter.update(loss.item())
        mask_meter.update(mask.mean().item())
        kl_meter.update(kl_divergence)
        kl_hard_meter.update(kl_divergence_hard)
        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())
        entropy_meter.update(mean_entropy)
        if (it + 1) % n_iter == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)
            logger.info("{}-x{}-s{}, {} | epoch:{}, iter: {}.  loss: {:.3f}. kl: {:.3f}. kl_hard:{:.3f}."
                        "LR: {:.3f}. Time: {:.2f}. Entropy: {:.2f}".format(
                args.dataset, args.n_labeled, args.seed, args.exp_dir, epoch, it + 1, loss_prop_meter.avg, kl_meter.avg,
                kl_hard_meter.avg, lr_log, t,entropy_meter.avg))

            epoch_start = time.time()
            bagsize = getattr(args, "bagsize", getattr(args, "bag_size", None))
            if bagsize is None:
                raise ValueError("请在 args 中提供 bagsize 或 bag_size")

            exp_dir_name = os.path.basename(os.path.normpath(args.exp_dir))

            os.makedirs(args.exp_dir, exist_ok=True)
            csv_name = f"{args.dataset}_{exp_dir_name}_{bagsize}.csv"
            csv_path = os.path.join(args.exp_dir, csv_name)

            is_new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["epoch", "time_sec"])
                writer.writerow([epoch, round(t, 2)])

            logger.info(f"CSV path -> {os.path.abspath(csv_path)}")
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
            if dataset in ["MNIST", "FashionMNIST", "KMNIST"]:
                ims = ims.permute(0, 2, 1, 3)

            out = model(ims)
            if isinstance(out, (tuple, list)):
                logits = out[0]  # 取第一个作为 logits
            else:
                logits = out
            loss = torch.nn.CrossEntropyLoss()(logits, lbs)

            loss_meter.update(loss.item())

            scores = torch.softmax(logits, dim=1)
            preds = scores.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbs.cpu().numpy())

            top1, top5 = accuracy(scores, lbs, (1, 2))
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

            if ema_model is not None:
                out = ema_model(ims)
                if isinstance(out, (tuple, list)):
                    ema_logits = out[0]  # 取第一个作为 logits
                else:
                    ema_logits = out

                ema_scores = torch.softmax(ema_logits, dim=1)
                ema_preds = ema_scores.argmax(dim=1)
                ema_all_preds.extend(ema_preds.cpu().numpy())
                ema_all_labels.extend(lbs.cpu().numpy())

                ema_top1, ema_top5 = accuracy(ema_scores, lbs, (1, 2))
                ema_top1_meter.update(ema_top1.item())


    return top1_meter.avg, ema_top1_meter.avg, top5_meter.avg, ema_top5_meter.avg, loss_meter.avg


def main():
    parser = argparse.ArgumentParser(description='DLLP Cifar Training')
    parser.add_argument('--root', default='./data', type=str, help='dataset directory')
    parser.add_argument('--wresnet-k', default=8, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default="CIFAR100",
                        help='number of classes in dataset')
    parser.add_argument('--n-classes', type=int, default=100,
                        help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=10,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=64,
                        help='train batch size of bag samples')
    parser.add_argument('--bagsize', type=int, default=16,
                        help='train bag size of samples')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=1024,
                        help='number of training images for each epoch')

    parser.add_argument('--eval-ema', default=False, help='whether to use ema model for evaluation')
    parser.add_argument('--ema-m', type=float, default=0.999)

    parser.add_argument('--lam-u', type=float, default=1.,
                        help='c oefficient of unlabeled loss')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random behaviors, no seed if negtive')

    parser.add_argument('--temperature', default=0.2, type=float, help='softmax temperature')
    parser.add_argument('--low-dim', type=int, default=64)
    parser.add_argument('--lam-c', type=float, default=1,
                        help='coefficient of contrastive loss')
    parser.add_argument('--lam-p', type=float, default=2,
                        help='coefficient of proportion loss')
    parser.add_argument('--contrast-th', default=0.8, type=float,
                        help='pseudo label graph threshold')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--queue_batch', type=float, default=5,
                        help='number of batches stored in memory bank')
    parser.add_argument('--exp-dir', default='DLLP', type=str, help='experiment id')
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
    dltrain_u, dataset_length, input_dim = get_train_loader(args.n_classes,
                                                            args.dataset, args.batchsize, args.bagsize, root=args.root,
                                                            method='DLLP',
                                                            supervised=False)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2, root=args.root)
    model, criteria_x, criteria_u, ema_model = set_model(args,input_dim)
    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))



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
            train_one_epoch(epoch, bagsize=args.bagsize, n_classes=args.n_classes, **train_args, samp_ran=samp_ran,
                            )

        top1, ema_top1, top5, ema_top5, loss_test = evaluate(model, ema_model, dlval,args.dataset)
        tb_logger.log_value('loss_prob', loss_prob, epoch)
        if (n_strong_aug == 0):
            tb_logger.log_value('guess_label_acc', 0, epoch)
        else:
            tb_logger.log_value('guess_label_acc', n_correct_u_lbs / n_strong_aug, epoch)
        tb_logger.log_value('test_acc', top1, epoch)
        tb_logger.log_value('mask', mask_mean, epoch)

        tb_logger.log_value('loss_test', loss_test, epoch)
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        if best_acc_5 < top5:
            best_acc_5 = top5
            best_epoch_5 = epoch
        logger.info(
            "Epoch {}.loss_test: {:.4f}. Acc: {:.4f}. Ema-Acc: {:.4f}. best_acc: {:.4f} in epoch{},Acc_5: {:.4f}.  best_acc_5: {:.4f} in epoch{},".
            format(epoch, loss_test, top1, ema_top1, best_acc, best_epoch, top5, best_acc_5, best_epoch_5))

        if top1 == best_acc:  # Check if current top1 accuracy is the best
            best_acc = top1
            best_epoch = epoch

            # Save the best model based on best accuracy
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'lr_scheduler': lr_schdlr.state_dict(),
                'prob_list': prob_list,
                'queue': {'queue_feats': queue_feats, 'queue_probs': queue_probs, 'queue_ptr': queue_ptr},
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(output_dir, 'best_acc.pth'))  # Save as 'best_acc.pth'


if __name__ == '__main__':
    main()