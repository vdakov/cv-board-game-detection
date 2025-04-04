from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.cluster import DBSCAN
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import urllib.request


def cluster_masks(anns, min_samples=3, eps=0.5):
    features = np.array([ann["area"] for ann in anns]).reshape(-1, 1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)

    for i, ann in enumerate(anns):
        ann["cluster"] = clustering.labels_[i]

    return anns


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam_checkpoint_path",
        help="Echo the path to the SAM checkpoint here.",
        type=str,
        default="board_segmentation/models/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--model_name",
        help="Echo the name of the model you want to use",
        type=str,
        default="vit_h",
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


def show_anns(original, anns, cluster=False, show_plots=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)

    if show_plots:
        _, ax = plt.subplots(1, 3, figsize=(15, 5))

    if cluster:
        sorted_anns = cluster_masks(sorted_anns)

    H, W = sorted_anns[0]["segmentation"].shape
    img = np.zeros((H, W, 4))  # Start with black image (RGBA)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m, :] = color_mask  #

    cluster_img = np.zeros((H, W, 4))
    if cluster:
        unique_clusters = set(
            ann["cluster"] for ann in sorted_anns if ann["cluster"] != -1
        )
        print("Estimated number of clusters: %d" % len(unique_clusters))
        print(
            "Estimated number of noise points: %d"
            % len(set(ann["cluster"] for ann in sorted_anns if ann["cluster"] == -1))
        )
        cluster_colors = {
            c: np.concatenate([np.random.random(3), [0.7]]) for c in unique_clusters
        }

        for ann in sorted_anns:
            cluster_label = ann["cluster"]
            if cluster_label != -1:  # Ignore outliers
                m = ann["segmentation"]
                cluster_img[m, :] = cluster_colors[cluster_label]

    if show_plots:
        ax[0].imshow(original)
        ax[1].imshow(img)
        ax[2].imshow(cluster_img)
        plt.axis("off")
        plt.show()

    return cluster_img


if __name__ == "__main__":
    args = get_args()
    im_folder = args.board_directory
    save_dir = args.save_dir
    show_plots = args.show_plots

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    sam_checkpoint = Path(args.sam_checkpoint_path)
    if not sam_checkpoint.is_file():
        print("Segment Anything Model not found. Downloading...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            sam_checkpoint,
        )
    sam = sam_model_registry[args.model_name](checkpoint=sam_checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Get all png and jpg files from the input directory
    image_files = [
        f
        for f in os.listdir(im_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"No image files found in {im_folder}")
        exit(1)

    for image_file in image_files:
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

        masks = mask_generator.generate(img)

        # Cluster the masks
        cluster_img = show_anns(img, masks, cluster=True, show_plots=show_plots)
        cluster_img = (cluster_img * 255).astype(np.uint8)
        cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)

        contours, _ = cv2.findContours(
            cluster_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        hexagons = []
        for contour in contours:
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )

            if len(approx) == 6:  # Hexagon has six sides
                hexagons.append(approx)

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

            save_path = os.path.join(output_subfolder, f"hex_{idx}.png")
            cv2.imwrite(save_path, cv2.cvtColor(hex_crop, cv2.COLOR_RGB2BGR))
