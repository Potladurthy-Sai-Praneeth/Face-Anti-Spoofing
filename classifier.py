import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from depth_map_model import FineTuneDepthAnything

class DepthClassifier(nn.Module):
    def __init__(self, hidden_dim=256, img_size=(252, 252)):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(16, 16)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, img_size[0], img_size[1])
            cnn_output_dim = self.feature_extractor(dummy_input).flatten(1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_dim, 2) )

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output
