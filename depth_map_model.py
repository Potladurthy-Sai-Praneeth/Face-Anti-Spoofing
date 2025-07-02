import torch
import torch.nn as nn
from transformers import AutoModelForDepthEstimation
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class FineTuneDepthAnything(nn.Module):
    def __init__(self):
        super(FineTuneDepthAnything, self).__init__()
        self.depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        for name,param in self.depth_anything.named_parameters():
            if 'head' in name or  'neck.fusion_stage.layers.3' in name or 'neck.fusion_stage.layers.2' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.depth_anything.config.output_hidden_states = True

    def forward(self, inp):
        outputs = self.depth_anything(inp)
        return outputs.predicted_depth.unsqueeze(1), outputs.hidden_states[-1][:,0,:]