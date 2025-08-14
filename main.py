#!/usr/bin/env python3
import os
import time
import datetime
import json
import math
import random
import argparse
from types import new_class
import numpy as np
from settings import parse_opts
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler

from thop import profile

from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from dataset.dataset import get_dataset_brats
from utils.utils import SequentialDistributedSampler
from trainer import train, evaluate, test

from monai.apps import DecathlonDataset
from monai.utils.misc import set_determinism
from monai.losses import FocalLoss, DiceLoss, DiceCELoss, DiceFocalLoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    CenterSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    SaveImage
)

from models.eoformer import EoFormer


def init_seeds(manual_seed):
    set_determinism(seed=manual_seed)
    os.environ['PYTHONHASHSEED'] = str(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameter: %.2fM, trainable parameter: %.2fM," % (total_num/1e6, trainable_num/1e6))


def main(args):
    
    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.distributed.init_process_group('nccl', world_size=len(args.gpu_id), rank=local_rank, timeout=datetime.timedelta(seconds=5400))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_seeds(args.manual_seed+local_rank)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        init_seeds(args.manual_seed)
    
    start = time.time()
    model_name = args.model

    save_folder = args.save_folder
    weight_save_folder = save_folder + f'/weights'
    gt_save_folder = save_folder + '/gt'
    pred_save_folder_dice = save_folder + '/predict/dice'
    pred_save_folder_hausdorff = save_folder + '/predict/hausdorff'
    
    if args.distributed:
        if local_rank == 0:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            if not os.path.exists(weight_save_folder):
                os.makedirs(weight_save_folder)
            if not os.path.exists(gt_save_folder):
                os.makedirs(gt_save_folder)
            if not os.path.exists(pred_save_folder_dice):
                os.makedirs(pred_save_folder_dice)
            if not os.path.exists(pred_save_folder_hausdorff):
                os.makedirs(pred_save_folder_hausdorff)
        torch.distributed.barrier()
    else:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if not os.path.exists(weight_save_folder):
            os.makedirs(weight_save_folder)
        if not os.path.exists(gt_save_folder):
            os.makedirs(gt_save_folder)
        if not os.path.exists(pred_save_folder_dice):
            os.makedirs(pred_save_folder_dice)
        if not os.path.exists(pred_save_folder_hausdorff):
            os.makedirs(pred_save_folder_hausdorff)
            
    modality_lst = sorted(args.modality)
    modality_num = len(modality_lst)
    
    train_crop_size = (args.crop_H, args.crop_W, args.crop_D)
    
    transform_brats20 = {
        'train': Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]), 
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=train_crop_size),
            RandSpatialCropd(keys=["image", "label"], roi_size=train_crop_size, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            ToTensord(keys=["image", "label"]),
            ]),
        
        'valid': Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
            ])
        }

    with open(f"{save_folder}/test_log.txt", "a") as f:

        f.write('Training start!\n')
        
        if args.dataset.lower() == 'brats':
            
            train_dataset, valid_dataset, test_dataset, train_list, valid_list, test_list = get_dataset_brats(
                data_path = args.data_path,
                json_file = args.json,
                transform=transform_brats20)
            
            if args.distributed:
                train_sampler = DistributedSampler(train_dataset)
                train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    sampler=train_sampler,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    prefetch_factor=4)
            
            else:
                train_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    prefetch_factor=4
                                )

            valid_loader = DataLoader(
                                    valid_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=args.num_workers
                                    )
            test_loader = DataLoader(
                                    test_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=args.num_workers
                                    )
            
        if args.distributed:
            if local_rank==0:
                print('train dataset len:', len(train_dataset))
                print('valid dataset len:', len(valid_dataset))
                print('test dataset len:', len(test_dataset))
                sub0 = train_dataset[3]
                print(sub0['image_meta_dict']['filename_or_obj'].split('/')[-1])
                print(f'train input shape: ', sub0['image'].shape, 'label shape:', sub0['label'].shape)
                sub1 = valid_dataset[3]
                print(sub1['image_meta_dict']['filename_or_obj'].split('/')[-1])
                print('valid input shape: ', sub1['image'].shape, 'label shape:', sub1['label'].shape)
                sub2 = test_dataset[3]
                print(sub2['image_meta_dict']['filename_or_obj'].split('/')[-1])
                print('test input shape: ', sub2['image'].shape, 'label shape:', sub2['label'].shape)
                
                print(' - - -'*20)
                print('device:', device)
                print(args)
                f.write(str(args) + '\n')
                print('Start Tensorboard with "tensorboard --logdir=/runs --port=6011" ')
                tb_writer = SummaryWriter(f'/{args.save_folder}/tb/', comment=f' {model_name}')
            torch.distributed.barrier()
        else:
            print('train dataset len:', len(train_dataset))
            print('valid dataset len:', len(valid_dataset))
            print('test dataset len:', len(test_dataset))
            sub0 = train_dataset[1]
            print(f'train input shape: ', sub0['image'].shape, 'label shape:', sub0['label'].shape)
            sub1 = valid_dataset[1]
            print('valid input shape: ', sub1['image'].shape, 'label shape:', sub1['label'].shape)
            sub2 = test_dataset[1]
            print('test input shape: ', sub2['image'].shape, 'label shape:', sub2['label'].shape)
            print(' - - -'*20)
            print('device:', device)
            print(args)
            f.write(str(args) + '\n')
            print('Start Tensorboard with "tensorboard --logdir=/runs --port=6011" ')
            tb_writer = SummaryWriter(f'/{args.save_folder}/tb/', comment=f' {model_name}')
        
        if args.model == 'eoformer':
            model = EoFormer(in_channels=modality_num, out_channels=args.n_seg_classes, drop_path=args.drop_path_rate)
        else:
            print(f"Error: no model {args.model}")
            exit(1)

        if args.distributed:
            model = nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank])
        else:
            model = nn.DataParallel(model.to(device), device_ids = args.gpu_id)

        if args.distributed:
            if local_rank==0:
                input = torch.randn(1, 4, 128, 128, 128).to(local_rank)
                flops, params = profile(model.module, (input,))
                print('Params = ' + str(params/1000**2) + 'M')
                print('FLOPs = ' + str(flops/1000**3) + 'G')
            torch.distributed.barrier()
        else:
            input = torch.randn(1, 4, 128, 128, 128).to(device)
            flops, params = profile(model.module, (input,))
            print('Params = ' + str(params/1000**2) + 'M')
            print('FLOPs = ' + str(flops/1000**3) + 'G')
        
        if args.distributed:
            if local_rank==0:
                get_parameter_number(model)
        else:
            get_parameter_number(model)
        
        pg = [p for p in model.parameters() if p.requires_grad]
        if args.optim == 'sgd':
            optimizer =  optim.SGD(pg, lr=args.learning_rate, momentum=0.99, weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer =  optim.Adam(pg, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif args.optim == "adamw":
            optimizer = torch.optim.AdamW(pg, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
        elif args.optim =='RMSprop':
            optimizer = optim.RMSprop(pg, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=args.weight_decay, momentum=0)

        if args.lr_scheduler == 'LambdaLR':
            lambda_cosine = lambda epoch_x: ((1 + math.cos(epoch_x * math.pi / args.n_epochs)) / 2) * (1 - args.learning_rate_fate) + 1e-2
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_cosine)
        elif args.lr_scheduler == 'StepLR':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif args.lr_scheduler == 'ExponentialLR':
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        elif args.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', patience=10, factor=0.2)
        elif args.lr_scheduler == "warmup_cosine":
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.n_epochs)

        if args.loss_function == 'CE':
            loss_fn = nn.CrossEntropyLoss()
        elif args.loss_function == 'Dice':
            loss_fn = DiceLoss(to_onehot_y=False, sigmoid=True)
        elif args.loss_function == 'DiceCE':
            loss_fn = DiceCELoss(to_onehot_y=False, sigmoid=True)
        elif args.loss_function == 'Focal':
            loss_fn = FocalLoss()
        elif args.loss_function == 'DiceFocal':
            loss_fn = DiceFocalLoss(to_onehot_y=False, sigmoid=True)
        
        start_epoch = args.start_epoch
        if args.resume:
            print('load checkpoint: ', weight_save_folder+ f'/checkpoint.pth')
            if os.path.isfile(weight_save_folder+ f'/checkpoint.pth'):
                checkpoint = torch.load(weight_save_folder+ f'/checkpoint.pth')
                start_epoch = checkpoint['epoch'] + 1
                best_dice = checkpoint['dice']
                best_miou = checkpoint['miou']
                model.module.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"=> loaded checkpoint path: {weight_save_folder+ '/checkpoint.pth'}, (epoch {checkpoint['epoch']})")
            else:
                print('=>checkpoint not exists!')
        else:
            if args.distributed:
                if local_rank == 0:
                    print("=> no checkpoint found")
                    best_dice = 0.5 if args.n_epochs > 50 else 0.0001
                    best_miou = 0.0
                    best_dice_epoch = 0
                    best_miou_epoch = 0
                    best_dice_cor_miou = 0.0
                    best_miou_cor_dice= 0.0
            else:
                print("=> no checkpoint found")
                best_dice = 0.5 if args.n_epochs > 50 else 0.01
                best_miou = 0.0
                best_dice_epoch = 0
                best_miou_epoch = 0
                best_dice_cor_miou = 0.0
                best_miou_cor_dice= 0.0
        
        min_epoch = 30 if args.n_epochs > 150 else 0

        model_inferer = partial(
            sliding_window_inference,
            roi_size=train_crop_size,
            sw_batch_size=args.sw_batch_size,
            predictor=model.module,
            overlap=args.inf_overlap,
        )
        
        nii_saver_gt = SaveImage(output_dir=gt_save_folder, output_postfix='gt', output_ext='.nii.gz', resample=True)
        nii_saver_pred_dice = SaveImage(output_dir=pred_save_folder_dice, output_postfix='pred', output_ext='.nii.gz', resample=True)
        nii_saver_pred_hausdorff = SaveImage(output_dir=pred_save_folder_hausdorff, output_postfix='pred', output_ext='.nii.gz', resample=True)
    
        for epoch in range(start_epoch, args.n_epochs+1):
            start_epoch = time.time()
            if args.distributed:
                train_sampler.set_epoch(epoch)
                train_loss = train(model, optimizer, loss_fn, train_loader, device=local_rank)
                if local_rank == 0:
                    print(f"epoch {epoch} train loss: {train_loss:.4f}")
            else:
                train_loss = train(model, optimizer, loss_fn, train_loader, device)
                print(f"epoch {epoch} train loss: {train_loss:.4f}")
            
            scheduler.step()
            if epoch % args.val_every == 0:
                if args.distributed:
                    if local_rank == 0:
                        valid_loss, valid_mean_dice, valid_dice_bg, valid_dice_lc_wm, valid_dice_lc_c, valid_dice_rc_wm, valid_dice_rc_c, valid_miou, valid_acc \
                            = evaluate(args, model, model_inferer, loss_fn, valid_loader, local_rank)
                        print(f"epoch {epoch} valid loss: {valid_loss:.4f}")
                        print(f"Dice mean: {valid_mean_dice:.4f}, bg: {valid_dice_bg:.4f}, lc_wm: {valid_dice_lc_wm:.4f}, lc_c: {valid_dice_lc_c:.4f}, rc_wm: {valid_dice_rc_wm:.4f}, rc_c: {valid_dice_rc_c:.4f}")
                        print(f"mIoU: {valid_miou:.4f}, Acc: {valid_acc:.4f}")
                        if epoch > min_epoch and valid_mean_dice > best_dice:
                            best_dice = valid_mean_dice
                            best_dice_cor_miou = valid_miou
                            best_dice_epoch = epoch
                            torch.save({
                                        'epoch': epoch,
                                        'dice': best_dice,
                                        'miou': best_dice_cor_miou,
                                        'state_dict': model.module.state_dict(),
                                        'optimizer': optimizer.state_dict()},
                                        weight_save_folder+ '/best_dice_model.pth')
                    torch.distributed.barrier()
                
                else:
                    valid_loss, valid_mean_dice, valid_dice_bg, valid_dice_lc_wm, valid_dice_lc_c, valid_dice_rc_wm, valid_dice_rc_c, valid_miou, valid_acc \
                        = evaluate(args, model, model_inferer, loss_fn, valid_loader, device)
                    print(f"epoch {epoch} valid loss: {valid_loss:.4f}")
                    print(f"Dice mean: {valid_mean_dice:.4f}, bg: {valid_dice_bg:.4f}, lc_wm: {valid_dice_lc_wm:.4f}, lc_c: {valid_dice_lc_c:.4f}, rc_wm: {valid_dice_rc_wm:.4f}, rc_c: {valid_dice_rc_c:.4f}")
                    print(f"mIoU: {valid_miou:.4f}, Acc: {valid_acc:.4f}")
                    if epoch > min_epoch and valid_mean_dice > best_dice:
                        best_dice = valid_mean_dice
                        best_dice_cor_miou = valid_miou
                        best_dice_epoch = epoch
                        torch.save({
                                    'epoch': epoch,
                                    'dice': best_dice,
                                    'miou': best_dice_cor_miou,
                                    'state_dict': model.module.state_dict(),
                                    'optimizer': optimizer.state_dict()},
                                    weight_save_folder+ f'/best_dice_model.pth')

            end_epoch = time.time()
            epoch_time = (end_epoch-start_epoch)/60
            if args.distributed:
                if local_rank==0:
                    print(f'epoch {epoch} time consuming: {epoch_time:.2f} min.\n')
            else:
                print(f'epoch {epoch} time consuming: {epoch_time:.2f} min.\n')

            tags = ["train loss", "valid loss", "valid dice", "valid miou", "learning_rate"]
            if args.distributed:
                if local_rank==0:
                    tb_writer.add_scalar('Loss/'+tags[0], train_loss, epoch)
                    tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            else:
                tb_writer.add_scalar('Loss/'+tags[0], train_loss, epoch)
                tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
                
        if args.distributed:
            if local_rank==0:
                print(f'best dice in validation set:{best_dice:.4f}, correspondence mIoU {best_dice_cor_miou:.4f}, in epoch: {best_dice_epoch}.')
                f.write(f'best dice in validation set:{best_dice:.4f}, correspondence mIoU {best_dice_cor_miou:.4f}, in epoch: {best_dice_epoch}.\n')
            torch.distributed.barrier()
        else:
            print(f'best dice in validation set:{best_dice:.4f}, correspondence mIoU {best_dice_cor_miou:.4f}, in epoch: {best_dice_epoch}.')
            f.write(f'best dice in validation set:{best_dice:.4f}, correspondence mIoU {best_dice_cor_miou:.4f}, in epoch: {best_dice_epoch}.\n')
        
        if args.distributed:
            if local_rank==0:
                print(' - - - - - test phase - - - - - ')
        else:
            print(' - - - - - test phase - - - - - ')
        best_dice_model_path = weight_save_folder+ f'/best_dice_model.pth'
        assert os.path.exists(best_dice_model_path), "cannot find {} file".format(best_dice_model_path)
        
        if args.distributed:
            if local_rank==0:
                dict_ = torch.load(best_dice_model_path, map_location='cuda:{}'.format(local_rank))
                model.module.load_state_dict(dict_['state_dict'], strict=False)
                print(f"dice model performance in valid set, dice: {dict_['dice']:.4f}, mIoU: {dict_['miou']:.4f}")
        else:
            dict_ = torch.load(best_dice_model_path)
            model.module.load_state_dict(dict_['state_dict'], strict=False)
            print(f"dice model performance in valid set, dice: {dict_['dice']:.4f}, mIoU: {dict_['miou']:.4f}")
        
        if args.distributed:
            if local_rank==0:
                test_mean_dice, test_dice_bg, test_dice_lc_wm, test_dice_lc_c, test_dice_rc_wm, test_dice_rc_c, test_miou, test_acc \
                = test(model, model_inferer, test_loader, nii_saver_gt, nii_saver_pred_dice, local_rank)
                print(f"Dice mean: {test_mean_dice:.4f}, bg: {test_dice_bg:.4f}, lc_wm: {test_dice_lc_wm:.4f}, lc_c: {test_dice_lc_c:.4f}, rc_wm: {test_dice_rc_wm:.4f}, rc_c: {test_dice_rc_c:.4f}")
                print(f"mIoU: {test_miou:.4f}, Acc: {test_acc:.4f}\n")
                f.write(f"Dice mean: {test_mean_dice:.4f}, bg: {test_dice_bg:.4f}, lc_wm: {test_dice_lc_wm:.4f}, lc_c: {test_dice_lc_c:.4f}, rc_wm: {test_dice_rc_wm:.4f}, rc_c: {test_dice_rc_c:.4f}\n")
                f.write(f"mIoU: {test_miou:.4f}, Acc: {test_acc:.4f}\n")
            torch.distributed.barrier()
        else:
            test_mean_dice, test_dice_bg, test_dice_lc_wm, test_dice_lc_c, test_dice_rc_wm, test_dice_rc_c, test_miou, test_acc \
            = test(model, model_inferer, test_loader, nii_saver_gt, nii_saver_pred_dice, device)
            print(f"Dice mean: {test_mean_dice:.4f}, bg: {test_dice_bg:.4f}, lc_wm: {test_dice_lc_wm:.4f}, lc_c: {test_dice_lc_c:.4f}, rc_wm: {test_dice_rc_wm:.4f}, rc_c: {test_dice_rc_c:.4f}")
            print(f"mIoU: {test_miou:.4f}, Acc: {test_acc:.4f}")
            f.write(f"Dice mean: {test_mean_dice:.4f}, bg: {test_dice_bg:.4f}, lc_wm: {test_dice_lc_wm:.4f}, lc_c: {test_dice_lc_c:.4f}, rc_wm: {test_dice_rc_wm:.4f}, rc_c: {test_dice_rc_c:.4f}\n")
            f.write(f"mIoU: {test_miou:.4f}, Acc: {test_acc:.4f}\n")
    f.close()

if __name__ == '__main__':
    
    opt = parse_opts()
    main(opt)