import torch
import torch.nn as nn
from depth_map_model import FineTuneDepthAnything
from classifier import DepthClassifier
from torch.utils.data import DataLoader
from dataloader import CustomDataset, collate_fn

class AntiSpoofing(nn.Module):
    def __init__(self,depth_model_path=None,classifier_path=None):
        super().__init__()
        assert depth_model_path is not None, 'Provide the path to depth map model'
        assert classifier_path is not None, 'Provide the path to classifier model'
        self.depth_model = FineTuneDepthAnything()
        self.depth_model.load_state_dict(torch.load(depth_model_path))

        self.classifier = DepthClassifier()
        self.classifier.load_state_dict(torch.load(classifier_path))

        self.depth_model.eval()
        self.classifier.eval()

    @torch.no_grad
    def forward(self,inp):
        depth_maps, _ =  self.depth_model(inp)
        classifier_ops = self.classifier(depth_maps)
        return depth_maps,classifier_ops