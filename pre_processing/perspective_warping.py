
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

def perspective_warp(img, bbox_coords):
    h, w = img.shape[:2]  # Keep original height and width

    margin_x = w // 5  # Margin for X-coordinates to keep within image bounds
    margin_y = h // 5  # Margin for Y-coordinates

    # Randomized x-coordinates for the top and bottom edges within safe margins
    new_x_top_left = np.random.randint(margin_x // 2, margin_x)
    new_x_top_right = np.random.randint(w - margin_x, w - margin_x // 2)
    new_x_bottom_left = np.random.randint(margin_x // 2, margin_x)
    new_x_bottom_right = min(new_x_bottom_left + (new_x_top_right - new_x_top_left), w - 1)

    # Y-axis transformations, keeping within bounds
    y_length = np.random.randint(h // 3, h // 2)
    y_top_left_coord = np.random.randint(margin_y, h // 2)

    # Ensuring Y-coordinates remain within valid ranges
    y_top_right_coord = min(y_top_left_coord + np.random.randint(-20, 20), h - 1)
    y_bottom_left_coord = min(y_top_left_coord + y_length, h - 1)
    y_bottom_right_coord = min(y_bottom_left_coord + (y_top_right_coord - y_top_left_coord), h - 1)

    # Ensure the new bounding box stays inside the image
    new_coords = np.array([
        [max(0, min(new_x_top_left, w - 1)), max(0, min(y_top_left_coord, h - 1))],
        [max(0, min(new_x_bottom_left, w - 1)), max(0, min(y_bottom_left_coord, h - 1))],
        [max(0, min(new_x_top_right, w - 1)), max(0, min(y_top_right_coord, h - 1))],
        [max(0, min(new_x_bottom_right, w - 1)), max(0, min(y_bottom_right_coord, h - 1))]
    ], dtype=np.float32)

    # Compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(bbox_coords, new_coords)

    # Apply perspective warp without resizing
    out_img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)  # Avoid black edges

    return out_img, new_coords


def perspective_warp_all(input_dir, output_dir, bbox_csv):

    bbox_output = {}

    for img_path in os.listdir(input_dir):
        if img_path.endswith(".png"):
            img = cv2.imread(os.path.join(input_dir, img_path)) 
            bbox_coords_df = bbox_csv.iloc[[2]] 
            x_min, x_max, y_min, y_max = bbox_coords_df['x_min'], bbox_coords_df['x_max'], bbox_coords_df['y_min'], bbox_coords_df['y_max']
            bbox_coords = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]], dtype=np.float32)
            

            warped_img, new_bbox = perspective_warp(img, bbox_coords)

            # Save the warped image
            warped_img_path = os.path.join(output_dir, img_path)
            cv2.imwrite(warped_img_path, warped_img)
            
            # Store new bbox coordinates
            bbox_output[img_path] = new_bbox.tolist()
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
