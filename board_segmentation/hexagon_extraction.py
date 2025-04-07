from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path
import urllib.request

import torch
from board_segmentation.segmentation import (
    extract_hexagon_contours,
    filter_for_hexagons,
    segment_all,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam_checkpoint_path",
        help="Echo the path to the SAM checkpoint here.",
        type=str,
        default="board_segmentation/models/sam_vit_b_01ec64.pth",
    )
    parser.add_argument(
        "--model_name",
        help="Echo the name of the model you want to use",
        type=str,
        default="vit_b",
    )
    parser.add_argument(
        "--board_directory",
        help="Echo the path to the directory with the board images",
        type=str,
        default="board_segmentation/data/input",
    )
    parser.add_argument(
        "--save_dir",
        help="Echo the path to the directory where you want your hexagons saved",
        type=str,
        default="board_segmentation/data/output",
    )
    parser.add_argument(
        "--show_plots",
        help="Whether to display visualizations during processing",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    return args


def load_segment_anything(sam_checkpoint_path, model_name):

    sam_checkpoint = Path(sam_checkpoint_path)
    if not sam_checkpoint.is_file():
        print("Segment Anything Model not found. Downloading...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            sam_checkpoint,
        )
    sam = sam_model_registry[model_name](checkpoint=sam_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    return mask_generator


def extract_and_save_masks_directory(mask_generator, image_files, im_folder, save_dir):
    centers = []
    for image_file in image_files:

        img = load_image(im_folder, image_file, save_dir)
        masks = segment_all(mask_generator, img)
        cluster_img = filter_for_hexagons(img, masks, show_plots=show_plots)
        hexagons = extract_hexagon_contours(cluster_img)

        if show_plots:
            # Draw detected hexagons for visualization
            result = img.copy()
            cv2.drawContours(result, hexagons, -1, (0, 255, 0), 3)

            # Plot the result
            plt.figure(figsize=(10, 5))
            plt.imshow(result, cmap="gray")
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f"Detected Hexagons in {image_file}")
            plt.axis("off")
            plt.show()

        for idx, hexagon in enumerate(hexagons):
            x, y, w, h = cv2.boundingRect(hexagon)  # Get bounding box
            hex_crop = img[y : y + h, x : x + w]  # Crop the region

            # Calculate the center of the hexagon
            M = cv2.moments(hexagon)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            centers.append((cx, cy))  # Store the center

            save_path = os.path.join(save_dir, f"hex_{idx}.png")
            cv2.imwrite(save_path, cv2.cvtColor(hex_crop, cv2.COLOR_RGB2BGR))

    return centers


def extract_single_image_hexagon(img, mask_generator, show_plots=False):
    centers = []
    masks = segment_all(mask_generator, img)
    cluster_img = filter_for_hexagons(img, masks, show_plots=show_plots)
    hexagons = extract_hexagon_contours(cluster_img)

    output = []

    for _, hexagon in enumerate(hexagons):
        x, y, w, h = cv2.boundingRect(hexagon)  # Get bounding box
        hex_crop = img[y : y + h, x : x + w]  # Crop the region
        pil_crop = Image.fromarray(cv2.cvtColor(hex_crop, cv2.COLOR_BGR2RGB))
        output.append(pil_crop)

        M = cv2.moments(hexagon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        centers.append((cx, cy))  # Store the center

    return output, centers


def load_image(im_folder, image_file, save_dir):
    print(f"Processing {image_file}...")
    image_path = os.path.join(im_folder, image_file)

    # Create output subfolder with the name of the image file (without extension)
    image_name = os.path.splitext(image_file)[0]
    output_subfolder = os.path.join(save_dir, image_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img


if __name__ == "__main__":
    args = get_args()
    im_folder = args.board_directory
    save_dir = args.save_dir
    show_plots = args.show_plots
    os.makedirs(save_dir, exist_ok=True)

    mask_generator = load_segment_anything(args.sam_checkpoint_path, args.model_name)

    # Get all png and jpg files from the input directory
    image_files = [
        f
        for f in os.listdir(im_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No image files found in {im_folder}")
        exit(1)

    extract_and_save_masks_directory(mask_generator, image_files, im_folder, save_dir)
