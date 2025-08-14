import os
import pdb
import json
from cv2 import log
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterator, List, Optional, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

# 混合精度训练
from torch.cuda.amp import autocast as autocast
from torch.cuda import amp

from utils.utils import calculate_metric

cal_mean_dice= DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True) # 返回一批数据的TC WT ET dice平均值, 若nan 返回0
cal_hausdorff = HausdorffDistanceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, percentile=95, get_not_nans=True) # 我们的0类不是背景，所以要包含bg(默认为第0维度)进去计算

post_sigmoid = Activations(sigmoid=True)
scaler = amp.GradScaler(enabled=True)


def train(model, optimizer, loss_fn, data_loader, device):
    model.train()
    n_ctr = 0
    n_loss = 0
    for step, batch_data in enumerate(data_loader):
        optimizer.zero_grad()
        images, labels = batch_data['image'], batch_data['label']
        images = images.to(device)
        labels = labels.to(device, dtype=torch.float32)
        with autocast(enabled=True):
            probs = model(images) # shape: [bs, n_classes, 24, 256, 256]
            loss = loss_fn(probs, labels)

        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        # for name, param in model.named_parameters():
        #     if param.grad == None:
        #         print(name)
        scaler.step(optimizer)
        scaler.update()
        n_loss += loss.item()
        n_ctr += 1
    return n_loss/n_ctr


def evaluate(args, model, model_inferer, loss_fn, data_loader, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        n_loss = 0
        sum_dice_avg = 0.0
        sum_miou = 0.0
        sum_acc = 0.0
        n_class = None
        sum_dice = None
        for step, batch_data in enumerate(data_loader):
            images, labels = batch_data['image'], batch_data['label']
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            with autocast(enabled=True):
                probs = model_inferer(images)
                loss = loss_fn(probs, labels)
            n_loss += loss.item()
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
            dpc, dice_avg, miou, acc = calculate_metric(pred_masks.cpu().numpy(), labels.cpu().numpy())
            if n_class is None:
                n_class = int(dpc.shape[0])
                sum_dice = np.zeros(n_class, dtype=np.float64)
            sum_dice += dpc.numpy()
            sum_dice_avg += dice_avg.item()
            sum_miou += miou.item()
            sum_acc += acc.item()
            n_ctr += 1
        dice_per_class = (sum_dice / max(n_ctr, 1)).tolist()
        dice_avg_mean = sum_dice_avg / max(n_ctr, 1)
        miou_mean = sum_miou / max(n_ctr, 1)
        acc_mean = sum_acc / max(n_ctr, 1)

    return n_loss/max(n_ctr,1), dice_avg_mean, dice_per_class[0], dice_per_class[1], dice_per_class[2], dice_per_class[3], dice_per_class[4], miou_mean, acc_mean


def test(model, model_inferer, data_loader, saver0, saver1, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        sum_dice_avg = 0.0
        sum_miou = 0.0
        sum_acc = 0.0
        n_class = None
        sum_dice = None
        for step, batch_data in enumerate(data_loader):
            images, labels = batch_data['image'], batch_data['label']
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)
            with autocast(enabled=True):
                probs = model_inferer(images)
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
            label = labels[0, 1]
            label[np.where(labels[0, 0] == 1)] = 2
            label[np.where(labels[0, 2] == 1)] = 3
            seg_img = pred_masks[0, 1]
            seg_img[np.where(pred_masks[0, 0] == 1)] = 2
            seg_img[np.where(pred_masks[0, 2] == 1)] = 3
            saver0(label)
            saver1(seg_img)
            dpc, dice_avg, miou, acc = calculate_metric(pred_masks.cpu().numpy(), labels.cpu().numpy())
            if n_class is None:
                n_class = int(dpc.shape[0])
                sum_dice = np.zeros(n_class, dtype=np.float64)
            print('dice:')
            print('bg:', dpc[0].item())
            print('lc_wm:', dpc[1].item())
            print('lc_c:', dpc[2].item())
            print('rc_wm:', dpc[3].item())
            print('rc_c:', dpc[4].item())
            sum_dice += dpc.numpy()
            sum_dice_avg += dice_avg.item()
            sum_miou += miou.item()
            sum_acc += acc.item()
            n_ctr += 1
        dice_per_class = (sum_dice / max(n_ctr, 1)).tolist()
        dice_avg_mean = sum_dice_avg / max(n_ctr, 1)
        miou_mean = sum_miou / max(n_ctr, 1)
        acc_mean = sum_acc / max(n_ctr, 1)

    return dice_avg_mean, dice_per_class[0], dice_per_class[1], dice_per_class[2], dice_per_class[3], dice_per_class[4], miou_mean, acc_mean


def inference(model, model_inferer, data_loader, saver, device='cuda'):
    
    model.eval()
    with torch.no_grad():
        n_ctr = 0
        
        for step, batch_data in enumerate(data_loader):
            images = batch_data['image']
            images = images.to(device)
            
            with autocast(enabled=True):
                probs = model_inferer(images)
            
            probs_sigmoid = post_sigmoid(probs)
            pred_masks = (probs_sigmoid >= 0.5).to(device, dtype=torch.float32)
            
            seg_img = pred_masks[0, 1]
            # seg_img[np.where(pred_masks[0, 1] == 1)] = 1
            seg_img[np.where(pred_masks[0, 0] == 1)] = 2
            seg_img[np.where(pred_masks[0, 2] == 1)] = 3
            saver(seg_img)
            n_ctr += 1
            
    return n_ctr