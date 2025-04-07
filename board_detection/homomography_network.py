import torch
import torch.nn as nn
import torch.nn.functional as F


class HomographyNet(nn.Module):
    def __init__(self, input_shape=(1, 128, 128), output_dim=9):
        """
        Args:
            input_shape (tuple): (C, H, W)
            output_dim (int): Number of parameters to predict (usually 9).
        """
        super(HomographyNet, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
        )

        # Compute the flattened feature size using a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # shape: (1, C, H, W)
            out = self.convolutions(dummy)
            flattened_size = out.view(1, -1).size(1)

        self.fully_connected = nn.Sequential(
            nn.Linear(flattened_size, 1024), nn.ReLU(), nn.Linear(1024, output_dim)
        )

    def forward(self, x):
        x = self.convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)

        # Append fixed tail to make a 3x3 matrix
        fixed_tail = (
            torch.tensor([0.0, 0.0, 1.0], device=x.device)
            .unsqueeze(0)
            .repeat(x.size(0), 1)
        )
        x = torch.cat([x[:, :-3], fixed_tail], dim=1)
        return x
