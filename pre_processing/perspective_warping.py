import cv2
import numpy as np
import os
import json
import pandas as pd
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Enter the path to the directory with the images on which perspective distortion will be applied.",
        type=str,
        default="pre_processing/data/output/synthetic_boards_blended",
    )
    parser.add_argument(
        "--output_dir",
        help="Input the path to the directory where the output of the distorted images will be stored.",
        type=str,
        default="pre_processing/data/output/perspective_distorted_boards",
    )
    parser.add_argument(
        "--bbox_json_dir",
        help="Input the path to the directory where the bounding boxes of the original boards are.",
        type=str,
        default="pre_processing/data/output/synthetic_boards_blended/bboxes.json",
    )
    args = parser.parse_args()
    return args


def perspective_warp(img, bbox_coords):
    h, w = img.shape[:2]  # Image height and width

    [x_min, y_min], _, [x_max, y_max], _ = bbox_coords

    angle_perturbation = np.random.uniform(-np.pi / 3, np.pi / 3)
    rotation_perturbation = np.random.uniform(-np.pi / 6, np.pi / 6)
    scaling_height = np.random.uniform(0.75, 1.25)
    scaling_width = np.random.uniform(0.75, 1.25)

    x_min_top = x_min
    x_min_bottom = x_min_top + np.cos(90 - angle_perturbation)
    x_max_top = x_min + scaling_width * (x_max - x_min)
    x_max_bottom = x_max_top + np.cos(90 - angle_perturbation)

    old_height = y_max - y_min
    new_height = scaling_height * old_height

    if new_height > old_height:
        y_min_perturbed = y_min - (new_height - old_height) // 2
        y_max_perturbed = y_max + (new_height - old_height) // 2
    else:
        y_min_perturbed = y_min + (new_height - old_height) // 2
        y_max_perturbed = y_max - (new_height - old_height) // 2

        # Generate random perturbations for each corner within the range
    perturbed_coords = np.float32(
        np.array(
            [
                [x_min_top, y_min_perturbed],
                [x_min_bottom, y_max_perturbed],
                [x_max_bottom, y_max_perturbed],
                [x_max_top, y_min_perturbed],
            ]
        )
    ).reshape(4, 2)

    # Rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(rotation_perturbation), -np.sin(rotation_perturbation)],
            [np.sin(rotation_perturbation), np.cos(rotation_perturbation)],
        ]
    )

    # # Rotate each coordinate
    center = np.mean(
        perturbed_coords, axis=0
    )  # Find the center point of your quadrilateral
    perturbed_coords_centered = (
        perturbed_coords - center
    )  # Translate to origin for rotation
    perturbed_coords = (
        rotation_matrix @ perturbed_coords_centered.T
    ).T  # Apply rotation correctly
    perturbed_coords += center

    # Ensure the perturbed points are within image boundaries
    perturbed_coords[:, 0] = np.clip(perturbed_coords[:, 0], 0, w - 1)
    perturbed_coords[:, 1] = np.clip(perturbed_coords[:, 1], 0, h - 1)

    perturbed_coords = np.array(perturbed_coords, dtype=np.float32)
    bbox_coords = np.array([coord for coord in bbox_coords], dtype=np.float32)
    # print(bbox_coords, perturbed_coords)

    # Compute homography matrix
    matrix = cv2.getPerspectiveTransform(bbox_coords, perturbed_coords)

    # Apply homography warp
    out_img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return out_img, perturbed_coords, matrix


def perspective_warp_all(input_dir, output_dir, bbox_df):

    bbox_output = {}

    for img_path in os.listdir(input_dir):
        if img_path.endswith(".png"):
            img = cv2.imread(os.path.join(input_dir, img_path))
            bbox_coords = np.array(bbox_df[img_path])

            # Format within json
            # bbox_coords = np.float32([
            #     [x_min, y_min],
            #     [x_min, y_max],
            #     [x_max, y_max],
            #     [x_max, y_min],
            # ])

            warped_img, new_bbox, matrix = perspective_warp(img, bbox_coords)

            # Save the warped image
            warped_img_path = os.path.join(output_dir, img_path)
            cv2.imwrite(warped_img_path, warped_img)

            # Store new bbox coordinates
            bbox_output[img_path] = new_bbox.tolist()
            bbox_output[f"{img_path}_homography_matrix"] = matrix.tolist()
            print(f"Image {img_path} warped and saved")

    # Save bounding box coordinates as JSON
    with open(os.path.join(output_dir, "bbox_coordinates.json"), "w") as f:
        json.dump(bbox_output, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    bbox_json_dir = args.bbox_json_dir
    bbox_df = pd.read_json(bbox_json_dir)

    perspective_warp_all(input_dir, output_dir, bbox_df)
