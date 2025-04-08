from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np
import torch
from board_detection.homomography_network import HomographyNet


def get_warped_image_bounds(H, width, height):
    corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )

    corners = np.concatenate([corners, np.ones((4, 1))], axis=1)
    warped_corners = (H @ corners.T).T
    warped_corners /= warped_corners[:, 2].reshape(-1, 1)
    warped_corners = warped_corners[:, :2]

    x_coords = warped_corners[:, 0]
    y_coords = warped_corners[:, 1]

    min_x, max_x = int(np.floor(np.min(x_coords))), int(np.ceil(np.max(x_coords)))
    min_y, max_y = int(np.floor(np.min(y_coords))), int(np.ceil(np.max(y_coords)))

    return min_x, min_y, max_x, max_y


def perspective_correct_image(
    input,
    model_checkpoint_path,
    model_resolution=128,
    path_or_img="path",
    show_image=True,
):
    checkpoint = torch.load(model_checkpoint_path)
    model = HomographyNet((1, model_resolution, model_resolution))
    model.load_state_dict(checkpoint["best_model_state_dict"])

    # Define the transform pipeline
    input_transform_pipeline = transforms.Compose(
        [
            transforms.Resize((model_resolution, model_resolution)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
        ]
    )

    if path_or_img == "path":
        # Load the image using PIL
        img = Image.open(input).convert("RGB")
    elif path_or_img == "img":
        img = input  # Assuming PIL image as input

    H, W = img.size  # Use .size to get width and height

    s_x = model_resolution / W
    s_y = model_resolution / H
    S = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]])

    transformed_image = input_transform_pipeline(img)
    transformed_image = transformed_image.unsqueeze(0)

    H_model = model(transformed_image)
    H_model = H_model.squeeze().detach().numpy().reshape(3, 3)
    # H_mat = np.linalg.inv(S) @ H_model @ S
    inverse_H = np.linalg.inv(H_model)
    img_np = np.array(img)

    min_x, min_y, max_x, max_y = get_warped_image_bounds(inverse_H, W, H)

    pad_left = max(0, -min_x)
    pad_top = max(0, -min_y)
    pad_right = max(0, max_x - W)
    pad_bottom = max(0, max_y - H)

    # Pad the image
    padded_img = cv2.copyMakeBorder(
        img_np,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    corrected_img = cv2.warpPerspective(
        padded_img, inverse_H, (W + pad_left + pad_right, H + pad_top + pad_bottom)
    )

    if show_image:
        plt.figure(figsize=(6, 6))
        plt.title("Perspective Corrected Image")
        plt.imshow(corrected_img)
        plt.axis("off")
        plt.show()
    return Image.fromarray(corrected_img)
