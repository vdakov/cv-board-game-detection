import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from board_detection.homography_loss import PhotometricLoss
from board_detection.homography_dataset import HomographyDataset
from visualization import utils
from tqdm import tqdm


def show_perspective_corrected_board(model, image, label, show_image=True):
    matrix = model(image.unsqueeze(0))
    image = image.squeeze().cpu().detach().numpy()
    label, matrix = label.detach().cpu().numpy().reshape(
        3, 3
    ), matrix.detach().cpu().numpy().reshape(3, 3)

    (h, w) = image.shape[:2]
    transformed_image_prediction = cv2.warpPerspective(
        image.copy(), np.linalg.inv(matrix + 1e-6), (w, h)
    )

    print(matrix)
    print(label)
    

    if show_image:
        transformed_image_label = cv2.warpPerspective(
            image.copy(), np.linalg.inv(label), (w, h)
        )
        utils.compare_prediction_ground_truth(
            image, transformed_image_prediction, transformed_image_label
        )

    return transformed_image_prediction


def load_dataset(
    json_path, img_dir, input_transform_pipeline, label_transform_pipeline
):
    with open(json_path, "r") as f:
        data = json.load(f)

    dataset = HomographyDataset(
        json_path,
        img_dir,
        input_transform=input_transform_pipeline,
        output_transform=label_transform_pipeline,
    )
    return dataset


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    save_path="model.pth",
    resume_path=None,
):
    """
    Trains a PyTorch model, optionally saving and resuming training.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        device (str): Device to train on ('cuda' or 'cpu').
        save_path (str): Path to save the model.
        resume_path (str, optional): Path to resume training from.
    """

    model.to(device)
    train_losses = []
    val_losses = []
    start_epoch = 0
    best_val_loss = float("inf")
    best_model_state_dict = None  # To save the best model weights

    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses = checkpoint["train_losses"]
        val_losses = checkpoint.get("val_losses", [])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        best_model_state_dict = checkpoint.get("best_model_state_dict", None)
        print(f"Resuming training from epoch {start_epoch+1}")

    for epoch in range(start_epoch + 1, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

        for i, (inputs, labels) in enumerate(pbar):
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
            avg_loss = running_loss / (i + 1)

            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                if isinstance(criterion, PhotometricLoss):
                    loss = criterion(outputs, labels, inputs)
                else:
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f"Epoch [{epoch}/{num_epochs}], Validation Loss: {epoch_val_loss:.4f}")

        # Save checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state_dict = model.state_dict()  # Save the best model weights
            print(f"New best model found at epoch {epoch} with validation loss {best_val_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "best_model_state_dict": best_model_state_dict,
        }
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")

    return train_losses, val_losses, best_model_state_dict




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
