import os
import math
import pickle
import itertools
import numpy as np

import torch
import glob
import torch.nn as nn
from monai.data import CacheDataset, Dataset, DataLoader
from monai.losses import DiceLoss

import sys
sys.path.append('..')

import models as models
from utils import SmoothedValue
from .base_trainer import BaseTrainer
from data.med_transforms import get_vit_transform

import wandb

from collections import defaultdict

class nnUnetTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.model_name = "nnUnet"
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            
            if args.loss_fn == 'dec':
                self.loss_fn = diceCELoss()
            elif args.loss_fn == 'wdec':
                self.loss_fn = weightedDiceCELoss(args.loss_weights)
            elif args.loss_fn == 'gdec':
                self.loss_fn = generalizedDiceCELoss()
            elif args.loss_fn == 'focal':
                self.loss_fn = focalLoss()
            elif args.loss_fn == 'dice':
                self.loss_fn = diceLoss()
            elif args.loss_fn == 'bce':
                self.loss_fn = BCELoss()
            elif args.loss_fn == 'gdice':
                self.loss_fn = generalizedDiceLoss()
            elif args.loss_fn == 'gdicefocal':
                self.loss_fn = generalizedDiceFocalLoss()
            elif args.loss_fn == 'Tloss':
                self.loss_fn = TLoss(image_size=args.roi_x, device=self.device)
            else:
                raise ValueError("=> Loss function is still unknown")

            net_params = {
                "input_channels": args.in_channels,
                "base_num_features": args.base_num_features,
                "num_classes": args.out_channels,
                "num_pool": len(args.pool_op_kernel_sizes),
                "num_conv_per_stage": 2,
                "feat_map_mul_on_downscale": 2,
                "conv_op": nn.Conv2d,
                "norm_op": nn.BatchNorm2d,
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": nn.Dropout2d,
                "dropout_op_kwargs": {"p": 0, "inplace": True},
                "nonlin": nn.LeakyReLU,
                "nonlin_kwargs": {"negative_slope": 1e-2, "inplace": True},
                "deep_supervision": args.deep_supervision,
                "dropout_in_localization": False,
                "final_nonlin": lambda x: x,
                "pool_op_kernel_sizes": args.pool_op_kernel_sizes,
                "conv_kernel_sizes": args.conv_kernel_sizes,
                "upscale_logits": False,
                "convolutional_pooling": True,
                "convolutional_upsampling": True,
            }

            
            self.model = getattr(models, self.model_name)(**net_params)

            self.model = self.model.to(self.device)
            self.model = nn.DataParallel(self.model)
        
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")
        
    def build_optimizer(self):
        assert(self.model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)
        if args.loss_fn == 'Tloss':
            added_params = {
                "params": self.loss_fn.parameters(),
                "lr": args.student_lr,
                "betas": (args.beta1, args.beta2),
                "weight_decay": args.weight_decay,
                }
            self.optimizer.add_param_group(added_params)
    
    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args
            pkl_path = os.path.join(args.data_path, args.pkl_list)
            with open(pkl_path, 'rb') as file:
                loaded_dic = pickle.load(file)

            train_ds = []
            for dic_tr in loaded_dic['training']:
                dic_tr['image'] = os.path.join(args.data_path, dic_tr['image'])
                dic_tr['label_bml'] = os.path.join(args.data_path, dic_tr['label_bml'])
                dic_tr['label_bone'] = os.path.join(args.data_path, dic_tr['label_bone'])
                train_ds.append(dic_tr)

            val_ds = []
            for dic_vl in loaded_dic['validating']:
                dic_vl['image'] = os.path.join(args.data_path, dic_vl['image'])
                dic_vl['label_bml'] = os.path.join(args.data_path, dic_vl['label_bml'])
                dic_vl['label_bone'] = os.path.join(args.data_path, dic_vl['label_bone'])
                val_ds.append(dic_vl)


            train_transform = get_vit_transform(args, 'train')
            train_dataset = CacheDataset(train_ds,
                                          transform=train_transform,
                                          num_workers=args.workers,
                                          cache_num=len(loaded_dic['training']))

            val_transform = get_vit_transform(args, 'valide')
            val_dataset = Dataset(val_ds, transform=val_transform)
            
            self.dataloader = DataLoader(train_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         num_workers=self.workers,
                                         pin_memory=True,
                                         sampler=None,
                                         drop_last=True)
            
            self.iters_per_epoch = len(self.dataloader)
            
            self.val_dataloader = DataLoader(val_dataset, 
                                            batch_size=args.val_batch_size, 
                                            shuffle=True,
                                            num_workers=self.workers, 
                                            pin_memory=True, 
                                            sampler=None,
                                            drop_last=False)
            self.val_iters = len(self.val_dataloader)
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")

    def run(self):
        args = self.args
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            niters = self.epoch_train(epoch, niters)

            # evaluate after each epoch training
            if epoch == 0 or (epoch + 1) % args.eval_freq == 0:
                self.evaluate(epoch=epoch, niters=niters)
            

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.model
        optimizer = self.optimizer
        scaler = self.scaler
        loss_fn = self.loss_fn

        model.train()

        for i, input_batch in enumerate(train_loader):

            image = input_batch['image']
            if args.out_channels == 1:
                target = input_batch['label_bml']
            elif args.out_channels == 2 or args.out_channels == 3:
                target = input_batch['label']

            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            if self.device.type == "cuda":
                image = image.to(self.device)
                target = target.to(self.device)
                with torch.cuda.amp.autocast(True):
                    loss = self.train_class_batch(model, image, target, loss_fn)
                        
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log to the screen
            if i % args.print_freq == 0:
                if 'lr_scale' in optimizer.param_groups[0]:
                    last_layer_lr = optimizer.param_groups[0]['lr'] / optimizer.param_groups[0]['lr_scale']
                else:
                    last_layer_lr = optimizer.param_groups[0]['lr']

                if args.loss_fn == 'Tloss':
                    nu = torch.mean(loss_fn.nu)
                    print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"nu: {nu:05f} | "
                    #   f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {last_layer_lr:.05f} | "
                      # f"PeRate: {model.module.pe_rate:.05f} | "
                      f"Loss: {loss.item():.03f} | ")
                else:
                    print(f"Epoch: {epoch:03d}/{args.epochs} | "
                        f"Iter: {i:05d}/{self.iters_per_epoch} | "
                        f"Lr: {last_layer_lr:.05f} | "
                        f"Loss: {loss.item():.03f} | ")
                if args.rank == 0:
                    wandb.log(
                        {
                        "lr": last_layer_lr,
                        "Loss": loss.item(),
                        },
                        step=niters,
                    )

            niters += 1
        return niters


    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = cur_lr * param_group['lr_scale']
            else:
                param_group['lr'] = cur_lr    

    @staticmethod
    def train_class_batch(model, samples, target, loss_fn):
        outputs = model(samples)
        return loss_fn(outputs, target)
        
    @torch.no_grad()
    def evaluate(self, epoch, niters):
        args = self.args
        model = self.model
        val_loader = self.val_dataloader
  
        BCELoss_fn = torch.nn.BCEWithLogitsLoss()
        if args.out_channels == 1:
            DiceLoss_fn = DiceLoss(include_background=True, sigmoid=True)
        elif args.out_channels == 2 or args.out_channels == 3:
            DiceLoss_fn = DiceLoss(include_background=True, sigmoid=True, reduction='none')
        else:
            raise ValueError(f"Undefined out channel {args.out_channels}")

        meters = defaultdict(SmoothedValue)

        # switch to evaluation mode
        model.eval()

        for i, input_batch in enumerate(val_loader):

            image = input_batch['image']
            if args.out_channels == 1:
                target = input_batch['label_bml']
            elif args.out_channels == 2 or args.out_channels == 3:
                target = input_batch['label']
            
            if self.device.type == "cuda":
                image = image.to(self.device)
                target = target.to(self.device)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(image)
                ce_loss = BCELoss_fn(output, target)
                dice = 1.0- DiceLoss_fn(output, target)

            batch_size = image.size(0)
            if args.out_channels == 1:
                meters['bce_loss'].update(value=ce_loss.item(), n=batch_size)
                meters['dice'].update(value=dice.item(), n=batch_size)

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                        f"Iter: {i:05d}/{self.val_iters} | "
                        f"BCE_loss: {ce_loss.item():.03f} | "
                        f"Dice: {dice.item():.03f} | ")
            elif args.out_channels == 2 or args.out_channels == 3:
                meters['bce_loss'].update(value=ce_loss.item(), n=batch_size)
                meters['dice'].update(value=dice[:, 0, ...].mean().item(), n=batch_size)
                meters['dice_bone'].update(value=dice[:, 1, ...].mean().item(), n=batch_size)

                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                        f"Iter: {i:05d}/{self.val_iters} | "
                        f"BCE_loss: {ce_loss.item():.03f} | "
                        f"Dice: {dice[:, 0, ...].mean().item():.03f} | "
                        f"Dice Bone: {dice[:, 1, ...].mean().item():.03f} | ")
         
        
        if args.out_channels==1:
            print(f"==> Epoch {epoch:04d} test results: \n"
                    f"=> BCE_loss: {meters['bce_loss'].global_avg:.05f} \n"
                    f"=> Dice: {meters['dice'].global_avg:.05f} \n")
            
            dice_global = meters['dice'].global_avg
            mean_val = np.mean(sorted(self.val_score)[-20:])
            if dice_global > mean_val:
                self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/{args.run_name}_Dice{dice_global:.02f}_checkpoint_{epoch:04d}.pth.tar'
                    )
                del_files = sorted(glob.glob(f'{args.ckpt_dir}/{args.run_name}*.pth.tar'), key= lambda x: os.stat(x).st_mtime)[:int(-1*args.save_ckpt_num)]

                if len(del_files) > 0:
                    for del_file_path in del_files:
                        os.remove(del_file_path)

                
            self.val_score.append(dice_global)

            if args.rank == 0:
                wandb.log(
                    {
                        "Eval BCE loss": meters['bce_loss'].global_avg,
                        "Val Dice": meters['dice'].global_avg,
                    },
                    step=niters,
                )

        elif args.out_channels == 2 or args.out_channels == 3:
            print(f"==> Epoch {epoch:04d} test results: \n"
                    f"=> BCE_loss: {meters['bce_loss'].global_avg:.05f} \n"
                    f"=> Dice: {meters['dice'].global_avg:.05f} \n"
                    f"=> Dice Bone: {meters['dice_bone'].global_avg:.05f} \n")
            
            dice_global = meters['dice'].global_avg
            mean_val = np.mean(sorted(self.val_score)[-20:])
            if dice_global > mean_val:
                self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, 
                        is_best=False, 
                        filename=f'{args.ckpt_dir}/{args.run_name}_Dice{dice_global:.02f}_checkpoint_{epoch:04d}.pth.tar'
                    )
                del_files = sorted(glob.glob(f'{args.ckpt_dir}/{args.run_name}*.pth.tar'), key= lambda x: os.stat(x).st_mtime)[:int(-1*args.save_ckpt_num)]

                if len(del_files) > 0:
                    for del_file_path in del_files:
                        os.remove(del_file_path)

                
            self.val_score.append(dice_global)

            if args.rank == 0:
                wandb.log(
                    {
                        "Eval BCE loss": meters['bce_loss'].global_avg,
                        "Val Dice": meters['dice'].global_avg,
                        "Val Dice Bone": meters['dice_bone'].global_avg,
                    },
                    step=niters,
                )
