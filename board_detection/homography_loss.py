import torch
import torch.nn as nn
import torch.nn.functional as F

class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()


    def forward(self, H_pred, H_true, I):
        """
        Args:
            H_pred (torch.Tensor): Predicted homography matrix of shape (B, 9)
            H_true (torch.Tensor): Ground truth homography matrix of shape (B, 9)
            I (torch.Tensor): Original images of shape (B, C, H, W)
        Returns:
            torch.Tensor: Photometric loss (scalar)
        """
        batch_size, _, H, W = I.size()

        # Reshape 9-element vectors to 3x3 matrices
        H_pred = H_pred.view(batch_size, 3, 3)
        H_true = H_true.view(batch_size, 3, 3)

        # Invert the homography matrices
        H_pred_inv = torch.linalg.inv(H_pred)
        H_true_inv = torch.linalg.inv(H_true)

        def build_perspective_grid(H_inv, height, width):
            """
            Builds a perspective sampling grid given an inverted 3x3 homography.
            Args:
                H_inv (torch.Tensor): Inverted homography matrix of shape (B, 3, 3)
                height (int): Image height
                width (int): Image width
            Returns:
                torch.Tensor: Sampling grid of shape (B, height, width, 2) in normalized [-1, 1] coordinates.
            """
            device = H_inv.device

            # Create a meshgrid of pixel coordinates (in the original image coordinate system)
            xs = torch.linspace(0, width - 1, width, device=device)
            ys = torch.linspace(0, height - 1, height, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # grid_y: (H, W), grid_x: (H, W)

            # Flatten the grid and add homogeneous coordinate 1
            ones = torch.ones_like(grid_x)
            grid = torch.stack([grid_x, grid_y, ones], dim=0)  # shape (3, H, W)
            grid = grid.view(3, -1)  # shape (3, H*W)
            grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # shape (B, 3, H*W)

            # Apply the homography transformation: new_points = H_inv * grid
            new_points = torch.bmm(H_inv, grid)  # shape (B, 3, H*W)

            # Normalize by the third (homogeneous) coordinate
            new_points = new_points / new_points[:, 2:3, :]  # shape (B, 3, H*W)

            # Extract x and y coordinates
            x_new = new_points[:, 0, :]  # shape (B, H*W)
            y_new = new_points[:, 1, :]  # shape (B, H*W)

            # Normalize coordinates to [-1, 1] for grid_sample
            x_norm = (2 * x_new / (width - 1)) - 1
            y_norm = (2 * y_new / (height - 1)) - 1

            # Stack and reshape into grid form
            grid_norm = torch.stack([x_norm, y_norm], dim=2)  # shape (B, H*W, 2)
            grid_norm = grid_norm.view(batch_size, height, width, 2)
            return grid_norm

        # Build perspective grids using the full 3x3 transformation
        grid_pred = build_perspective_grid(H_pred_inv, H, W)
        grid_true = build_perspective_grid(H_true_inv, H, W)

        # Warp the image using the perspective grids
        I_pred_warped = F.grid_sample(I, grid_pred, mode='bilinear', padding_mode='zeros', align_corners=False)
        I_true_warped = F.grid_sample(I, grid_true, mode='bilinear', padding_mode='zeros', align_corners=False)

        # Compute the photometric loss (mean squared error between warped images)
        photometric_loss = F.mse_loss(I_pred_warped, I_true_warped)

        return photometric_loss