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
class CustomDataset(Dataset): 
    def __init__(self, path, img_size=(128, 128)):
        super(CustomDataset, self).__init__()
        self.client_names = []
        self.imposter_names =[]
        self.labels = []
        self.images = []
        self.ground_truth = []
        self.img_size = img_size
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.ToTensor()])
        
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
    
    scale_range = (0.7, 1.1)
        
    geometric_augs = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.4),
        K.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=scale_range,align_corners=True),
        data_keys=['input', 'mask'],  
        same_on_batch=False         
    )

    color_augs = K.AugmentationSequential(
        K.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.8)
    )

    
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
    
    loss_fn = CustomLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001,weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    num_epochs = 101
   
    # Main training loop
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    best_epoch_loss = float('inf')

    # Main training loop
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0
        
        for i, data in enumerate(train_loader, 0):
            binary_targets = data[1].to(device)
            augmented_image, depth_maps = geometric_augs(data[0].to(device), data[-1].to(device))
            inputs = color_augs(augmented_image)
            optimizer.zero_grad()
            
            outputs, binary_outputs = model(inputs)        
            loss = loss_fn(outputs, depth_maps, binary_outputs, binary_targets)*100

            running_accuracy += calculate_classification_accuracy(binary_outputs, binary_targets)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
            # plot_depth_maps(outputs, depth_maps, binary_targets)
        
        # Calculate average training metrics
        avg_train_loss = running_loss / len(train_loader)
        avg_train_accuracy = running_accuracy / num_batches
        
        scheduler.step(avg_train_loss)  # Use average loss for scheduler
        train_loss.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        if rank==0:
            print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f} , Training Accuracy: {avg_train_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Plot with correct binary labels
        # plot_depth_maps(outputs, depth_maps, binary_targets)
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:  
            if rank==0:      
                torch.save(model.module.state_dict(), f"fine_tuning_depth_anything_epoch_{epoch}.pth")
            model.eval()
            with torch.no_grad():
                running_loss_test = 0.0
                running_accuracy_test = 0.0
                num_batches_test = 0
                
                for i, data in enumerate(val_loader, 0):
                    binary_targets_test = data[1].to(device)
                    depth_maps_test = data[-1].to(device)
                    inputs_test = data[0].to(device)
                    
                    outputs_test, binary_outputs_test = model(inputs_test)
                    loss_test = loss_fn(outputs_test, depth_maps_test, binary_outputs_test, binary_targets_test)
                    
                    # Calculate validation classification accuracy
                    batch_accuracy_test = calculate_classification_accuracy(binary_outputs_test, binary_targets_test)
                    running_accuracy_test += batch_accuracy_test
                    
                    running_loss_test += loss_test.item()
                    num_batches_test += 1
                
                # Calculate average validation metrics
                avg_val_loss = running_loss_test / len(val_loader)
                avg_val_accuracy = running_accuracy_test / num_batches_test
                
                val_loss.append(avg_val_loss)
                val_accuracy.append(avg_val_accuracy)
                
                if rank==0:
                    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")
                    
                    # Plot with correct binary labels
                    # plot_depth_maps(outputs_test, depth_maps_test, binary_targets_test)
                    
                    # Save best model
                    if avg_val_loss < best_epoch_loss:
                        best_epoch_loss = avg_val_loss
                        torch.save(model.module.state_dict(), f"best_fine_tune_depth_anything_epoch_{epoch}.pth")
                        print(f"New best model saved with validation loss: {best_epoch_loss:.4f}")
    
    cleanup()

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

        self.hidden_state = self.depth_anything.backbone.encoder.layer[-1].mlp.fc2.out_features
        # self.hidden_state = 1024
        self.ffd_dim = 1024

        self.classifier = nn.Sequential(
                                        nn.Linear(self.hidden_state, self.ffd_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.4),
                                        nn.Linear(self.ffd_dim, self.ffd_dim),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(self.ffd_dim, self.ffd_dim//2),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(self.ffd_dim//2, 2)
                                        )
                        
    def forward(self, inp):
        outputs = self.depth_anything(inp, output_hidden_states=True)
        binary_predictions = self.classifier(outputs.hidden_states[-1][:,0,:])
        return outputs.predicted_depth.unsqueeze(1), binary_predictions

# %%
class ContrastDepthLoss(nn.Module):  
    def __init__(self,device,criterion):
        super(ContrastDepthLoss,self).__init__()
        self.kernel_filter = torch.tensor([
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
        ],dtype=torch.float32).to(device)
        
        self.criterion=criterion

    def contrast_depth_conv(self,input_tensor):
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
        contrast_depth = F.conv2d(input_expanded, weight=self.kernel_filter.unsqueeze(dim=1), groups=8)
        
        return contrast_depth

    def forward(self, out, label):     
        return self.criterion(self.contrast_depth_conv(out), self.contrast_depth_conv(label))

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, inputs, target):
        if inputs.dim()>2:
            inputs = inputs.view(inputs.size(0),inputs.size(1),-1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1,2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1,inputs.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(inputs,dim=1).gather(1,target).view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type()!=inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        
        if self.size_average: 
            return loss.mean()
            
        return loss.sum()

class CustomLoss(nn.Module):
    def __init__(self, device, lambda_focal=2.0, lambda_depth=1.0, lambda_blank=1.0):
        super(CustomLoss, self).__init__()
        self.device = device
        self.lambda_focal = lambda_focal
        self.lambda_depth = lambda_depth
        self.lambda_blank = lambda_blank

        self.focal_loss = FocalLoss()

        self.depth_criterion = nn.SmoothL1Loss(reduction='mean')
        self.contrast_loss = ContrastDepthLoss(self.device, self.depth_criterion)

        self.blanking_criterion = nn.SmoothL1Loss(reduction='mean')

    def forward(self, pred_depth, gt_depth, pred_binary, gt_binary):
        focal_loss = self.focal_loss(pred_binary, gt_binary)

        real_mask = (gt_binary == 1)
        spoof_mask = (gt_binary == 0)

        loss_depth = torch.tensor(0.0, device=self.device)
        if torch.any(real_mask):
            real_pred_depth = pred_depth[real_mask]
            real_gt_depth = gt_depth[real_mask]
            
            smooth_loss_real = self.depth_criterion(real_pred_depth, real_gt_depth)
            contrast_loss_real = self.contrast_loss(real_pred_depth, real_gt_depth)
            loss_depth = smooth_loss_real + contrast_loss_real
            
        loss_blank = torch.tensor(0.0, device=self.device)
        if torch.any(spoof_mask):
            spoof_pred_depth = pred_depth[spoof_mask]
            blank_target = torch.zeros_like(spoof_pred_depth)
            loss_blank = self.blanking_criterion(spoof_pred_depth, blank_target)

        total_loss = (self.lambda_focal * focal_loss + 
                      self.lambda_depth * loss_depth + 
                      self.lambda_blank * loss_blank)
        return total_loss

# %%
def plot_depth_maps(outputs, ground_truth, binary):
    index = {0: None, 1: None}
    sorted_labels_to_check = sorted(torch.unique(binary).cpu().numpy())
    
    for label in sorted_labels_to_check:
        if label not in [0, 1]:
            continue 
        
        matches = (binary == label).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            index[label] = matches[0].item()

    present_labels = [label for label, idx in index.items() if idx is not None]
    
    num_plots = len(present_labels) * 2

    if num_plots == 0:
        print("No samples with label 0 or 1 found in this batch to plot.")
        return

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * len(present_labels), 5), squeeze=False)
    axes = axes.flatten() 

    ax_counter = 0
    for label in present_labels:
        data_idx = index[label]

        # Plot Prediction
        pred_map = outputs[data_idx].cpu().detach().squeeze().numpy()
        axes[ax_counter].imshow(pred_map)
        axes[ax_counter].set_title(f'Prediction (Label {label})')
        axes[ax_counter].axis('off')
        ax_counter += 1

        # Plot Ground Truth
        gt_map = ground_truth[data_idx].cpu().detach().squeeze().numpy()
        axes[ax_counter].imshow(gt_map)
        axes[ax_counter].set_title(f'Ground Truth (Label {label})')
        axes[ax_counter].axis('off')
        ax_counter += 1
        
    plt.tight_layout()
    plt.show()

def calculate_classification_accuracy(binary_outputs, binary_targets):
    """Calculate binary classification accuracy"""
    predictions = torch.argmax(binary_outputs, dim=1)
    correct = (predictions == binary_targets).float()
    accuracy = correct.mean().item()
    return accuracy

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