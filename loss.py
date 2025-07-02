import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class ContrastDepthLoss(nn.Module):  
    def __init__(self,criterion):
        super(ContrastDepthLoss,self).__init__()
        kernel = torch.tensor([
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
        ],dtype=torch.float32)

        kernel = kernel.unsqueeze(dim=1)
        self.register_buffer("kernel_filter", kernel)
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
        input_expanded = input_tensor.expand(-1, 8, -1, -1)

        # Perform depthwise convolution using the defined kernel filters
        return F.conv2d(input_expanded, weight=self.kernel_filter, groups=8)

    def forward(self, out, label):     
        return self.criterion(self.contrast_depth_conv(out), self.contrast_depth_conv(label) )

class FocalLoss(nn.Module):
    def __init__(self, device,gamma=2, alpha=0.3, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.device = device
        if isinstance(alpha,(float,int)): 
            self.alpha = torch.tensor([alpha,1-alpha],device=device)
        if isinstance(alpha,list): 
            self.alpha = torch.tensor(alpha,device=device)
        self.size_average = size_average

    def forward(self, inputs, target):
        target = target.view(-1,1)

        logpt = F.log_softmax(inputs,dim=1).gather(1,target).view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()
    
class CustomLoss(nn.Module):
    def __init__(self, device, lambda_focal=1.0, lambda_depth=20.0, lambda_blank=0.5):
        super(CustomLoss, self).__init__()
        self.device = device
        # self.lambda_focal = lambda_focal
        self.lambda_depth = lambda_depth
        self.lambda_blank = lambda_blank
        # self.focal_loss = FocalLoss(self.device)
        self.depth_criterion = nn.SmoothL1Loss(reduction='mean')
        self.contrast_loss = ContrastDepthLoss(self.depth_criterion)

    def forward(self, pred_depth, gt_depth, gt_binary, lambda_depth=None, lambda_blank=None): # pred_binary, lambda_focal=None,
        if lambda_depth is not None and lambda_blank is not None: #lambda_focal is not None and
            self.lambda_depth = lambda_depth
            self.lambda_blank = lambda_blank
        real_mask = (gt_binary == 1)
        spoof_mask = ~real_mask

        loss_depth = pred_depth.new_tensor(0.0) 
        loss_blank = pred_depth.new_tensor(0.0) 

        if torch.any(real_mask):
            real_pred_depth = pred_depth[real_mask]
            real_gt_depth = gt_depth[real_mask]

            smooth_loss_real = self.depth_criterion(real_pred_depth, real_gt_depth)
            contrast_loss_real = self.contrast_loss(real_pred_depth, real_gt_depth)
            loss_depth = smooth_loss_real + contrast_loss_real
            
        if torch.any(spoof_mask):
            spoof_pred_depth = pred_depth[spoof_mask]
            blank_target = gt_depth[spoof_mask]            
            loss_blank = self.depth_criterion(spoof_pred_depth, blank_target) #+ contrast_loss_blank 

        total_loss = self.lambda_depth * loss_depth + self.lambda_blank * loss_blank #self.lambda_focal * focal_loss + 
        return total_loss
