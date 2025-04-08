from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import matplotlib.pyplot as plt
import os
import cv2
from pathlib import Path
import urllib.request
import json
import numpy as np

import torch
from segmentation import (
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
        default="board_segmentation/data/models/sam_vit_b_01ec64.pth",
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


def extract_and_save_masks_directory(
    mask_generator, image_files, im_folder, save_dir, show_plots=False
):
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
    hexagons = extract_hexagon_contours(cluster_img, show_plots=show_plots)

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


def load_image(im_folder, image_file, save_dir, make_dir=True):
    print(f"Processing {image_file}...")
    image_path = os.path.join(im_folder, image_file)

    # Create output subfolder with the name of the image file (without extension)
    image_name = os.path.splitext(image_file)[0]
    if make_dir:
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

    evaluation_results = {
        "image_names": [],
        "tile_counts": [],
        "avg_tile_sizes_percent": [],
        "total_images": len(image_files),
        "expected_tile_count": 19,  # Ideal number for Catan
    }
    # Process each image
    for image_file in image_files:
        print(f"Evaluating segmentation for {image_file}...")
        img = load_image(im_folder, image_file, save_dir, make_dir=False)
        image_area = img.shape[0] * img.shape[1]

        masks = segment_all(mask_generator, img.copy())
        cluster_img = filter_for_hexagons(img.copy(), masks, show_plots=False)
        if cluster_img is None:
            tiles_detected = 0
            avg_tile_size_percent = 0
        else:
            hexagons = extract_hexagon_contours(cluster_img, show_plots=False)
            tiles_detected = len(hexagons)
            if tiles_detected > 0:
                hexagon_areas = [
                    cv2.contourArea(hex_contour) for hex_contour in hexagons
                ]
                avg_tile_size = sum(hexagon_areas) / len(hexagon_areas)
                avg_tile_size_percent = (avg_tile_size / image_area) * 100
            else:
                avg_tile_size_percent = 0
        # Load individual info
        evaluation_results["image_names"].append(image_file)
        evaluation_results["tile_counts"].append(tiles_detected)
        evaluation_results["avg_tile_sizes_percent"].append(avg_tile_size_percent)

        print(f"  - Tiles detected: {tiles_detected}/19")
        print(f"  - Average tile size: {avg_tile_size_percent:.2f}% of image")
    # Load summary info
    evaluation_results["summary"] = {
        "mean_tile_count": np.mean(evaluation_results["tile_counts"]),
        "median_tile_count": np.median(evaluation_results["tile_counts"]),
        "std_tile_count": np.std(evaluation_results["tile_counts"]),
        "mean_tile_size_percent": np.mean(evaluation_results["avg_tile_sizes_percent"]),
        "median_tile_size_percent": np.median(
            evaluation_results["avg_tile_sizes_percent"]
        ),
        "std_tile_size_percent": np.std(evaluation_results["avg_tile_sizes_percent"]),
    }
    plt.figure(figsize=(10, 5))
    tile_count_data = evaluation_results["tile_counts"]
    max_count = (
        max(tile_count_data)
        if tile_count_data
        else evaluation_results["expected_tile_count"] + 1
    )
    min_count = min(tile_count_data) if tile_count_data else 0
    # First histogram - tile counts
    bins = np.arange(min_count - 0.5, max_count + 1.5, 1.0)
    plt.hist(tile_count_data, bins=bins, alpha=0.7, rwidth=0.85, edgecolor="black")
    plt.axvline(
        x=evaluation_results["expected_tile_count"],
        color="r",
        linestyle="--",
        label=f"Ideal ({evaluation_results['expected_tile_count']} tiles)",
    )
    plt.title("Distribution of Detected Tile Counts")
    plt.xlabel("Number of Tiles Detected")
    plt.ylabel("Frequency")
    plt.grid(axis="both", alpha=0.75)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "tile_count_histogram.png"))
    # Second histogram - tile sizes
    plt.figure(figsize=(10, 5))
    tile_size_data = evaluation_results["avg_tile_sizes_percent"]
    if tile_size_data:
        max_size = max(tile_size_data)
        bins = np.arange(0, max_size + 0.05, 0.05)
    else:
        bins = 15
    plt.hist(tile_size_data, bins=bins, alpha=0.7, rwidth=0.85, edgecolor="black")
    plt.title("Distribution of Average Tile Sizes")
    plt.xlabel("Average Tile Size (% of Image Area)")
    plt.ylabel("Frequency")
    plt.grid(axis="both", alpha=0.75)
    plt.savefig(os.path.join(save_dir, "tile_size_histogram.png"))
    # Save data to JSON
    with open(os.path.join(save_dir, "segmentation_evaluation_results.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)
    # Final print
    print(f"Results saved to {save_dir}")
    print("\nSummary Statistics:")
    print(f"Mean Tile Count: {evaluation_results['summary']['mean_tile_count']:.2f}/19")
    print(
        f"Mean Tile Size: {evaluation_results['summary']['mean_tile_size_percent']:.2f}% of image"
    )
    if show_plots:
        plt.show()
    else:
        plt.close("all")
