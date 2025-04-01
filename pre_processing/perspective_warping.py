
import cv2 
import numpy as np 
import os
import json
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help= "Enter the path to the directory with the images on which perspective distortion will be applied.", type=str, default="data/full/mined_synthetic_boards_blended")
    parser.add_argument("--output_dir", help="Input the path to the directory where the output of the distorted images will be stored.", type=str, default="data/full/perspective_distorted_boards")
    parser.add_argument("--bbox_csv_dir", help="Input the path to the directory where the bounding boxes of the original boards are.", type=str, default="data/full/mined_synthetic_boards_blended/bboxes.csv")
    args = parser.parse_args()
    return args

def perspective_warp(img, bbox_coords, max_perturbation=30):
    h, w = img.shape[:2]  # Image height and width
    x_min, x_max, y_min, y_max = bbox_coords
    print(bbox_coords)


    angle_perturbation = np.random.uniform(-np.pi/6, np.pi/6)
    rotation_perturbation = np.random.uniform(-np.pi/6, np.pi/6)
    scaling_height = np.random.uniform(0.75, 1.25)
    scaling_width = np.random.uniform(0.75, 1.25)

    x_min_top = x_min
    x_min_bottom = x_min_top + np.cos(90 - angle_perturbation) 
    x_max_top = x_min + scaling_width * (x_max - x_min)
    x_max_bottom = x_max_top + np.cos(90 - angle_perturbation) 
    
    old_height = y_max - y_min
    new_height = scaling_height * old_height
    print(new_height, old_height)
    if new_height > old_height: 
        y_min_perturbed = y_min - (new_height - old_height) // 2
        y_max_perturbed = y_max + (new_height - old_height) // 2
    else: 
        y_min_perturbed = y_min + (new_height - old_height) // 2
        y_max_perturbed = y_max - (new_height - old_height) // 2

        # Generate random perturbations for each corner within the range
    perturbed_coords = np.float32([
        [x_min_top, y_min_perturbed], [x_max_top, y_min_perturbed], [x_min_bottom, y_max_perturbed], [x_max_bottom, y_max_perturbed]
    ])

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(rotation_perturbation), -np.sin(rotation_perturbation)],
        [np.sin(rotation_perturbation), np.cos(rotation_perturbation)]
    ])

    # Rotate each coordinate
    center = np.mean(perturbed_coords, axis=0)  # Find the center point of your quadrilateral
    perturbed_coords_centered = perturbed_coords - center  # Translate to origin for rotation
    perturbed_coords = np.dot(perturbed_coords_centered, rotation_matrix.T)  # Apply rotation
    perturbed_coords += center
    
    # Ensure the perturbed points are within image boundaries
    perturbed_coords[:, 0] = np.clip(perturbed_coords[:, 0], 0, w - 1)
    perturbed_coords[:, 1] = np.clip(perturbed_coords[:, 1], 0, h - 1)

    # Compute homography matrix
    matrix = cv2.getPerspectiveTransform(bbox_coords, perturbed_coords)

    # Apply homography warp
    out_img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return out_img, perturbed_coords, matrix

def perspective_warp_all(input_dir, output_dir, bbox_csv):

    bbox_output = {}

    for img_path in os.listdir(input_dir):
        if img_path.endswith(".png"):
            img = cv2.imread(os.path.join(input_dir, img_path)) 
            bbox_coords_df = bbox_csv[bbox_csv["image_name"] == img_path] 
            print(bbox_coords_df)
            x_min, x_max, y_min, y_max = bbox_coords_df['x_min'], bbox_coords_df['x_max'], bbox_coords_df['y_min'], bbox_coords_df['y_max']
            bbox_coords = np.array([x_min, x_max, y_min, y_max], dtype=int)
            

            warped_img, new_bbox, matrix = perspective_warp(img, bbox_coords)

            # Save the warped image
            warped_img_path = os.path.join(output_dir, img_path)
            cv2.imwrite(warped_img_path, warped_img)
            
            # Store new bbox coordinates
            bbox_output[img_path] = new_bbox.tolist()
            bbox_output["homography_matrix"] = matrix
            print(f"Image {img_path} warped and saved")

    # Save bounding box coordinates as JSON
    with open(os.path.join(output_dir, "bbox_coordinates.json"), "w") as f:
        json.dump(bbox_output, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    bbox_csv_dir = args.bbox_csv_dir
    bbox_csv = pd.read_csv(bbox_csv_dir)

    perspective_warp_all(input_dir, output_dir, bbox_csv)
