# %%
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

# %%
def setup_ddp(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set the device for this process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

# %%
# class CustomDataset(Dataset): 
#     def __init__(self, path, img_size=(128, 128)):
#         super(CustomDataset, self).__init__()
#         self.images = []
#         self.labels = []
#         self.img_size = img_size
#         self.transform = transforms.Compose([ transforms.Resize(img_size), transforms.ToTensor()])
#         self.path = path
#         self.num_channels = 1
        
#         for folder in os.listdir(self.path):
#             label = 1 if 'client' in folder else 0
#             for image in os.listdir(os.path.join(self.path, folder)):
#                 if image.endswith('.jpg') or image.endswith('.png'):
#                     img_path = os.path.join(self.path, folder, image)
#                     self.images.append(img_path)
#                     self.labels.append(label)
        
#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = Image.open(self.images[idx]).convert("RGB")

#         if self.transform:
#             img= self.transform(img)
            
#         return img, self.labels[idx]


class CustomDataset(Dataset): 
    def __init__(self, path, img_size=(128, 128)):
        super(CustomDataset, self).__init__()
        self.client_names = []
        self.imposter_names =[]
        self.labels = []
        self.images = []
        self.ground_truth = []
        self.img_size = img_size
        self.transform = transforms.Compose([ transforms.Resize(img_size), transforms.ToTensor()])
        self.path = path
        self.num_channels = 1

        for folder in os.listdir(self.path):
            if folder == 'Processed_client':
                self.client_names.extend(os.listdir(os.path.join(self.path, folder)))
            if folder == 'Processed_imposter':
                self.imposter_names.extend(os.listdir(os.path.join(self.path, folder)))
        
        for image in self.client_names:
            img_path = os.path.join(self.path, 'Processed_client', image)
            self.images.append(img_path)
            self.labels.append(1)
            gt_path = os.path.join(self.path, 'gt_client', image)
            self.ground_truth.append(gt_path)
        
        for image in self.imposter_names:
            img_path = os.path.join(self.path, 'Processed_imposter', image)
            self.images.append(img_path)
            self.labels.append(0)
            gt_path = os.path.join(self.path, 'gt_imposter', image)
            self.ground_truth.append(gt_path)
       
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        gt = Image.open(self.ground_truth[idx]).convert("L")

        if self.transform:
            img= self.transform(img)
            gt = self.transform(gt)
            
        return img, self.labels[idx], gt

def collate_fn(batch):
  images, labels, gt = zip(*batch)
  images = torch.stack(images, 0)
  labels = torch.tensor(labels)
  gt = torch.stack(gt, 0)
  
  return images, labels, gt

# %%
def main_worker(rank, world_size,train_path, val_path):
    """Main training function for each process"""
    # **NEW: Setup DDP**
    setup_ddp(rank, world_size)
    
    scale_range = (0.8, 1.2)
    
    gpu_augmentation_pipeline = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomAffine(degrees=20, translate=None, scale=scale_range),
        K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.8)
    ).to(rank) 
    
    img_size = (252,252)
    batch_size = 128  #  This is per-GPU batch size**

    device = torch.device(f"cuda:{rank}")

    train_dataset = CustomDataset(train_path,img_size=img_size)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  
        num_workers=multiprocessing.cpu_count()//world_size,  
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler  
    )

    val_dataset = CustomDataset(val_path,img_size=img_size)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=multiprocessing.cpu_count()//world_size, 
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=val_sampler 
    )
    
    model = FineTuneDepthAnything().to(device)
    model = DDP(model, device_ids=[rank])
    
    kernel_filter = torch.tensor([
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ],dtype=torch.float32).to(device)
    
    # large_depth_map = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf").to(device)
    # large_depth_map.eval()
    # large_depth_map_compiled = torch.compile(large_depth_map,mode="max-autotune")
    
    criterion = nn.SmoothL1Loss()
    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    num_epochs = 101
    
    train_loss = []
    val_loss = []
    best_epoch_loss = float('inf')

    # Main training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            binary_labels = data[1].to(device)
            inputs = data[0].to(device)
            depth_maps = data[-1].to(device)

            inputs_aug = gpu_augmentation_pipeline(inputs)
            optimizer.zero_grad()
            
            outputs = model(inputs_aug)

            # with torch.no_grad():
            #     out = large_depth_map_compiled(inputs_aug)
            #     depth_maps = out.predicted_depth.unsqueeze(1)
            
            labels = get_labels(depth_maps, binary_labels)
            loss = customLoss(criterion, mse_criterion, outputs, labels, kernel_filter)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if rank == 0:
            torch.save(model.module.state_dict(), "fine_tuning_depth_anything.pth")
        
        scheduler.step(running_loss)
        train_loss.append(running_loss)
        
        if rank == 0:
            print(f"Epoch {epoch+1}, Training Loss: {running_loss} and lr is {optimizer.param_groups[0]['lr']}")
            plot_depth_maps(outputs, labels, binary_labels)

        if (epoch + 1) % 5 == 0:
            val_sampler.set_epoch(epoch)
            
            model.eval()
            with torch.no_grad():
                running_loss_test = 0.0
                for i, data in enumerate(val_loader, 0):
                    binary_labels_test = data[1].to(device)
                    inputs_aug_test = gpu_augmentation_pipeline(data[0].to(device))
                    depth_maps_test = data[-1].to(device)
                    outputs_test = model(inputs_aug_test)

                    # out = large_depth_map_compiled(inputs_aug_test)
                    # depth_maps_test = out.predicted_depth.unsqueeze(1)

                    labels_test = get_labels(depth_maps_test, binary_labels_test)
                    loss_test = customLoss(criterion, mse_criterion, outputs_test, labels_test, kernel_filter)
                    running_loss_test += loss_test.item()

                val_loss.append(running_loss_test)
                
                if rank == 0:
                    print(f"Validation Loss: {running_loss_test}")
                    plot_depth_maps(outputs_test, labels_test, binary_labels_test)

                    if running_loss_test < best_epoch_loss:
                        best_epoch_loss = running_loss_test
                        torch.save(model.module.state_dict(), "best_fine_tuning_depth_anything.pth")
    
    cleanup()

# %%
class CDC(nn.Module):
    '''
    This class performs central difference convolution (CDC) operation. First the normal convolution is performed and then the difference convolution is performed. The output is the difference between the two is taken.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(CDC, self).__init__()
        self.bias= bias
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.theta = theta
        self.padding = padding
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding if kernel_size==3 else 0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)
        # if conv.weight is (out_channels, in_channels, kernel_size, kernel_size),
        # then the  self.conv.weight.sum(2) will return (out_channels, in_channels,kernel_size)
        # and self.conv.weight.sum(2).sum(2) will return (out_channels,n_channels)
        kernel_diff = self.conv.weight.sum(2).sum(2)
        # Here we are adding extra dimensions such that the kernel_diff is of shape (out_channels, in_channels, 1, 1) so that convolution can be performed.
        kernel_diff = kernel_diff[:, :, None, None]
        out_diff = F.conv2d(x, weight=kernel_diff, bias=self.bias, stride=self.stride, padding=0, groups=self.groups)
        return out_normal - self.theta * out_diff

# %%
class FineTuneDepthAnything(nn.Module):
    def __init__(self):
        super(FineTuneDepthAnything, self).__init__()
        self.depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        for name,param in self.depth_anything.named_parameters():
            if 'head' in name or 'neck.fusion_stage.layers.2.residual_layer' in name or 'neck.fusion_stage.layers.3' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                        
    def forward(self, inp):
        return self.depth_anything(inp).predicted_depth.unsqueeze(1)

# %%
def contrast_depth_conv(input_tensor,kernel_filter):
    """
    Compute contrast depth using depthwise convolution.
    
    Parameters:
    - input_tensor: A tensor of shape (N, C, H, W), expected to be (N, 1, 32, 32)
    - device: The device (CPU/GPU) the tensors should be processed on
    
    Returns:
    - A tensor of shape (N, 8, H, W) representing the contrast depth
    """
    # Expand the input tensor to have 8 channels to match the number of kernel filters
    input_expanded = input_tensor.squeeze(1).unsqueeze(dim=1).expand(-1, 8, -1, -1)
    
    # Perform depthwise convolution using the defined kernel filters
    contrast_depth = F.conv2d(input_expanded, weight=kernel_filter.unsqueeze(dim=1), groups=8)
    
    return contrast_depth

# %%
def customLoss(criterion,mse_criterion,predictions,labels,kernel_filter):
    smooth_loss = criterion(predictions, labels)
    contrast_pred = contrast_depth_conv(predictions,kernel_filter)
    contrast_label = contrast_depth_conv(labels,kernel_filter)
    contrast_loss = mse_criterion(contrast_pred, contrast_label)
    return (0.3*smooth_loss + contrast_loss) * 100

# %%
def plot_depth_maps(outputs, labels, binary):
    # Initialize the index dictionary with None values
    index = {0: None, 1: None}
    
    # Try to find the first occurrence of 0 and 1, if they exist
    for label in [0, 1]:
        matches = (binary == label).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            index[label] = matches[0].item()
    
    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    
    # Plot for label 0 if it exists
    if index[0] is not None:
        ax[0].imshow(outputs[index[0]].cpu().detach().squeeze(0).squeeze(0).numpy())
        ax[0].set_title('Predictions for label 0')
        ax[1].imshow(labels[index[0]].cpu().detach().squeeze(0).numpy())
        ax[1].set_title('Ground truth for label 0')
    else:
        ax[0].set_visible(False)
        ax[1].set_visible(False)
    
    # Plot for label 1 if it exists
    if index[1] is not None:
        ax[2].imshow(outputs[index[1]].cpu().detach().squeeze(0).squeeze(0).numpy())
        ax[2].set_title('Predictions for label 1')
        ax[3].imshow(labels[index[1]].cpu().detach().squeeze(0).numpy())
        ax[3].set_title('Ground truth for label 1')
    else:
        ax[2].set_visible(False)
        ax[3].set_visible(False)
    
    plt.show()

def get_labels(inputs, label):
    mask = label.view(-1, 1, 1, 1) == 1
    return torch.where(mask, inputs, torch.zeros_like(inputs))

# %%
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