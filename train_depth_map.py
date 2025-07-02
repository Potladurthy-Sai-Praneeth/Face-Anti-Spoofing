import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import os
import sys
from torch.utils.data import DataLoader, Dataset
import math
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from transformers import pipeline
from PIL import Image
from torchvision import models
import torchvision.transforms.functional as TF
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time
from torch.profiler import profile,record_function, ProfilerActivity
import kornia
import kornia.augmentation as K
import multiprocessing
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

from dataloader import CustomDataset, collate_fn
from depth_map_model import FineTuneDepthAnything
from classifier import DepthClassifier
from loss import *
from utils import *
from config import CONFIG


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size,train_path, val_path):
    setup_ddp(rank, world_size)

    geometric_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.4),
        K.RandomAffine(degrees=CONFIG['rotation'], translate=CONFIG['translation'], scale=CONFIG['scale_range'],align_corners=False),
        data_keys=['input', 'mask'],  
        same_on_batch=False         
    )

    color_augs = K.AugmentationSequential(
        K.ColorJitter(brightness=CONFIG['brightness'], contrast=CONFIG['contrast'], saturation=CONFIG['saturation'], hue=CONFIG['hue'], p=CONFIG['probability'])
    )



    device = torch.device(f"cuda:{rank}")

    train_dataset = CustomDataset(train_path,img_size=CONFIG['img_size'])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,  
        num_workers=multiprocessing.cpu_count()//world_size,  
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler  
    )

    val_dataset = CustomDataset(val_path,img_size=CONFIG['img_size'])
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False, 
        num_workers=multiprocessing.cpu_count()//world_size, 
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=val_sampler 
    )

    model = FineTuneDepthAnything().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    train_loss = []
    val_loss = []
    best_epoch_loss = float('inf')
    lambda_depth = CONFIG['lambda_depth']
    lambda_blank = CONFIG['lambda_blank']

    loss_fn = CustomLoss(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)


    for epoch in range(CONFIG['num_epochs']):
        train_sampler.set_epoch(epoch)  
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0

        # Fixed approach curriculum learning
        if epoch == 5:
            lambda_depth = 10.0
            lambda_blank = 1.0
        
        if epoch == 15:
            lambda_depth = 5.0
            lambda_blank = 1.0
        
        if epoch == 30:
            lambda_depth = 2.0
            lambda_blank = 1.0
        
        for i, data in enumerate(train_loader, 0):
            binary_targets = data[1].to(device)
            augmented_image, depth_maps = geometric_augs(data[0].to(device), data[-1].to(device))
            inputs = color_augs(augmented_image)
            optimizer.zero_grad()
            
            outputs, _ = model(inputs)        
            loss = loss_fn(outputs, depth_maps,binary_targets,lambda_depth=lambda_depth,lambda_blank=lambda_blank) * 100 
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = running_loss / num_batches
        
        scheduler.step(avg_train_loss) 
        train_loss.append(avg_train_loss)

        # plot_depth_maps(outputs, depth_maps, binary_targets)

        if rank==0:
            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}") # Training Accuracy: {avg_train_accuracy:.4f},
        
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:  
            if rank==0:      
                torch.save(model.module.state_dict(), f"fine_tuning_depth_anything_epoch_{epoch+1}.pth")
            model.eval()
            with torch.no_grad():
                running_loss_test = 0.0
                running_accuracy_test = 0.0
                num_batches_test = 0
                
                for i, data in enumerate(val_loader, 0):
                    binary_targets_test = data[1].to(device)
                    depth_maps_test =  data[-1].to(device)
                    inputs_test = data[0].to(device)
                    
                    outputs_test, _ = model(inputs_test)
                    loss_test = loss_fn(outputs_test, depth_maps_test, binary_targets_test, lambda_depth=1.0,lambda_blank=1.0)*100
                    running_loss_test += loss_test.item()
                    num_batches_test += 1
                    
                # Calculate average validation metrics
                avg_val_loss = running_loss_test / num_batches_test                
                val_loss.append(avg_val_loss)
                
                if rank==0:
                    print(f"Validation Loss: {avg_val_loss:.4f}")
                    # Save best model
                    if avg_val_loss < best_epoch_loss:
                        best_epoch_loss = avg_val_loss
                        torch.save(model.module.state_dict(), f"best_fine_tune_depth_anything.pth")
                        print(f"New best model saved with validation loss: {best_epoch_loss:.4f}")
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Distributed Depth Anything Fine-tuning')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the training dataset')
    parser.add_argument('--val_path', type=str, required=True, help='Path to the validation dataset')
    args = parser.parse_args()
    train_path = args.train_path
    val_path = args.val_path

    world_size = torch.cuda.device_count() 
    print(f"Using {world_size} GPUs for training")
    
    if world_size > 1:
        torch.multiprocessing.spawn(main_worker, args=(world_size,train_path,val_path), nprocs=world_size, join=True)
    else:
        main_worker(0, 1,train_path,val_path)
