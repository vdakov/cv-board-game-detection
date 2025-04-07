import os
import json
import cv2
import random

# Paths
data_dir = "pre_processing/data/output/perspective_distorted_boards"
json_file = os.path.join(data_dir, "bbox_coordinates.json")
output_dir = "board_detection/data/input"
train_dir = os.path.join(output_dir, "train")
val_dir = os.path.join(output_dir, "val")
train_images_dir = os.path.join(train_dir, "images")
val_images_dir = os.path.join(val_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")
val_labels_dir = os.path.join(val_dir, "labels")
data_yaml_path = os.path.join(output_dir, "data.yaml")

# Ensure output directories exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Load annotations
with open(json_file, "r") as f:
    annotations = json.load(f)

# Shuffle and split data
image_names = list(annotations.keys())
image_names = [name for name in image_names if "matrix" not in name]
random.shuffle(image_names)
split_idx = int(0.8 * len(image_names))
train_images = image_names[:split_idx]
val_images = image_names[split_idx:]


def process_images(image_list, images_output_dir, labels_output_dir):
    for img_name in image_list:
        img_name_no_ext = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_output_dir, f"{img_name_no_ext}.txt")
        img_src_path = os.path.join(data_dir, img_name)
        img_dest_path = os.path.join(images_output_dir, img_name)

        # Load the image using OpenCV to get its dimensions
        img = cv2.imread(img_src_path)

        if img is None:
            print(f"Warning: Could not read image {img_name}")
            continue

        img_height, img_width, _ = img.shape  # height, width, channels
        bbox = annotations[img_name]

        with open(label_path, "w") as label_file:
            # Extract corner points
            x1, y1 = bbox[0]  # top-left
            x2, y2 = bbox[1]  # bottom-left
            x3, y3 = bbox[2]  # top-right
            x4, y4 = bbox[3]  # bottom-right

            # Find bounding box (x_min, y_min, x_max, y_max)
            x_min = min(x1, x2, x3, x4)
            y_min = min(y1, y2, y3, y4)
            x_max = max(x1, x2, x3, x4)
            y_max = max(y1, y2, y3, y4)

            # YOLO format: (class_id, x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            # Normalize to YOLO format
            x_center /= img_width
            y_center /= img_height
            width /= img_width
            height /= img_height

            # Write to file
            label_file.write(
                f"0 {x_center} {y_center} {width} {height}\n"
            )  # Assuming class name "

        # Copy image to corresponding folder
        cv2.imwrite(img_dest_path, img)


# Process train and validation sets
process_images(train_images, train_images_dir, train_labels_dir)
process_images(val_images, val_images_dir, val_labels_dir)

# Create data.yaml file
data_yaml = f"""
train: {train_images_dir}
val: {val_images_dir}
nc: 1
names: ['Catan-Board']
"""


with open(data_yaml_path, "w") as f:
    f.write(data_yaml)

print(f"{len(train_images)} training images and {len(val_images)} validation images.")
