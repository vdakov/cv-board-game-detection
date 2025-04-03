import json
import os
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import torch 
from board_detection.homography_loss import PhotometricLoss
from data.torch_data_loaders.homography_dataset import HomographyDataset
from visualization import utils
from tqdm import tqdm

def show_perspective_corrected_board(model, image, label, show_image=True):
    matrix = model(image.unsqueeze(0))
    image = image.squeeze().detach().numpy()
    label, matrix = label.detach().numpy().reshape(3, 3), matrix.detach().numpy().reshape(3, 3)

    (h, w) = image.shape[:2]
    transformed_image_prediction = cv2.warpPerspective(image.copy(), np.linalg.inv(matrix) , (w, h))


    if show_image:
        transformed_image_label = cv2.warpPerspective(image.copy(), np.linalg.inv(label), (w, h))
        utils.compare_prediction_ground_truth(image, transformed_image_prediction, transformed_image_label)

    return transformed_image_prediction


def load_dataset(json_path, img_dir, input_transform_pipeline, label_transform_pipeline):
    with open(json_path, 'r') as f:
        data = json.load(f)

    dataset = HomographyDataset(json_path, img_dir, input_transform=input_transform_pipeline, output_transform=label_transform_pipeline)
    return dataset 


def train_model(model, train_loader, criterion, optimizer, num_epochs, device, save_path="model.pth", resume_path=None):
    """
    Trains a PyTorch model, optionally saving and resuming training.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the model.
        resume_path (str, optional): Path to resume training from.
    """

    model.to(device)
    train_losses = []
    start_epoch = 0

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_losses = checkpoint['train_losses']
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch+1}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            if isinstance(criterion, PhotometricLoss):
                loss = criterion(outputs, labels, inputs)
            else:
                loss = criterion(outputs, labels)
            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")



    return train_losses

def calculate_test_loss(model, test_loader, criterion, device):
    test_loss = 0
    running_loss = 0
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        if isinstance(criterion, PhotometricLoss):
            loss = criterion(outputs, labels, inputs)
        else:
            loss = criterion(outputs, labels)
        running_loss += loss.item()

    test_loss = running_loss / len(test_loader.dataset)
    return test_loss 


