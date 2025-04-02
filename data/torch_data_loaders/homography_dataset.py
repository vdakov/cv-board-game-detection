import os
from torch.utils.data import *
from PIL import Image
import json
import torch

class HomographyDataset(Dataset):
    def __init__(self, json_path, img_dir, transform=None):
        with open(json_path, 'r') as f:
            self.homography_matrix_labels = json.load(f)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.homography_matrix_labels) // 2

    def __getitem__(self, idx):
        img_name = f"canvas_image_{idx}.png"
        img_path = os.path.join(self.img_dir, img_name)


        image = Image.open(img_path).convert("RGB")

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        
        # Flatten the homography matrix into a single label tensor
        matrix = self.homography_matrix_labels[f"{img_name}_homography_matrix"]
        label = torch.tensor([item for row in matrix for item in row], dtype=torch.float32)
        
        return image, label
