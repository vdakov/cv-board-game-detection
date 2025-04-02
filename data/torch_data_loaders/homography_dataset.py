import os
from torch.utils.data import *
from PIL import Image
import json
import torch
import numpy as np
from torchvision import transforms

class HomographyDataset(Dataset):
    def __init__(self, json_path, img_dir, input_transform=None, output_transform=None):
        with open(json_path, 'r') as f:
            self.homography_matrix_labels = json.load(f)
        self.img_dir = img_dir
        self.input_transform = input_transform
        self.output_transform = output_transform

    def __len__(self):
        return len(self.homography_matrix_labels) // 2

    def __getitem__(self, idx):
        img_name = f"canvas_image_{idx}.png"
        img_path = os.path.join(self.img_dir, img_name)


        image = Image.open(img_path).convert("RGB")

        # Apply transform if provided
        if self.input_transform:
            image = self.input_transform(image)
        
        # Flatten the homography matrix into a single label tensor
        matrix = self.homography_matrix_labels[f"{img_name}_homography_matrix"]
        if self.output_transform:
            label = self.output_transform(matrix)

        
        return image, label

class HomographyOutputTransform:
    def __init__(self, old_size, new_size):
        self.old_size = old_size
        self.new_size = new_size

    def __call__(self, homography_matrix):
        homography_numpy = np.array(homography_matrix, dtype=np.float32)


        s_x = self.old_size[0] / self.new_size[0]
        s_y = self.old_size[1] / self.new_size[1]

        S = np.array([[s_x, 0, 0],
                      [0, s_y, 0],
                      [0, 0, 1]])

        H_new = homography_numpy @ S
        return H_new

