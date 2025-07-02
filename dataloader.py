import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom dataset for loading images and ground truth depth maps
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
