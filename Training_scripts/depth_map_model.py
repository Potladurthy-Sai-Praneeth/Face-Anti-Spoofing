import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation,AutoConfig


class FineTuneDepthAnything(nn.Module):
    '''
    A class to fine-tune the Depth-Anything model for depth estimation. 
    The model is loaded from the Hugging Face model hub and only the last few layers are trained.
    Args :
        device : torch.device
            The device on which the model is loaded.
        load_trained : bool
            Whether to load the trained model or not. It is a flag which prevents the model from downloading the pre-trained when we have fine-tuned and obtained the new weights.
        model_path : str
            The path to the trained model.
    Returns :   
        depth_anything : torch.tensor
            The depth estimation of the given input image.
    
    The input is a 3-channel RGB image and the output is a 1-channel depth map.
    '''
    def __init__(self, device,load_trained=False,model_path=None):
        super(FineTuneDepthAnything, self).__init__()
        if load_trained:
            config = AutoConfig.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            self.depth_anything = AutoModelForDepthEstimation.from_config(config)
            state_dict = torch.load(model_path, map_location=device)
                
            # Adjust keys in the state dictionary to match the model's keys
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("depth_anything.", "")
                new_state_dict[new_key] = value

            # Load the adjusted state dictionary into the model
            self.depth_anything.load_state_dict(new_state_dict)
        else:
            self.depth_anything = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
            for name,param in self.depth_anything.named_parameters():
                if 'head' in name or 'neck.fusion_stage.layers.2.residual_layer' in name or 'neck.fusion_stage.layers.3' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        self.depth_anything = self.depth_anything.to(device)
                
    def forward(self, inp):
        # print(f'inp shape: {inp.shape}')
        return self.depth_anything(inp).predicted_depth.unsqueeze(1)