import argparse

import cv2
from matplotlib import transforms
import torch
import keras
import pickle
from ultralytics import YOLO
from board_detection.homomography_network import HomographyNet
from board_piece_classification.hexagon_prediction import predict_image
from board_detection.yolo_extraction import board_detection_step
from board_detection import perspective_correct_image
from PIL import Image
from board_segmentation.hexagon_extraction import (
    extract_single_image_hexagon,
    load_segment_anything,
)
import pytesseract
import os
import numpy as np
from PIL import Image
import os
import numpy as np
import json
from visualization.board_visualization import (
    standard_positions,
    adjacency_map,
    visualize_board,
    HEX_SIZE,
    vert,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        help="Input the path to your image here.",
        type=str,
        default="data/input/canvas_image_0.png",
    )
    parser.add_argument(
        "--hexagon_detector_path",
        help="Input the path to the model that detects hexagons here.",
        type=str,
        default="board_piece_classification/data/models/tile_detector_hexagons.keras",
    )
    parser.add_argument(
        "--hexagon_label_encoder_path",
        help="Input the path to the label encoder associated with the hexagon detector here.",
        type=str,
        default="board_piece_classification/data/models/label_encoder_hexagons.pkl",
    )
    parser.add_argument(
        "--output_path",
        help="Path to save the assembled board JSON file.",
        type=str,
        default="data/output",
    )
    parser.add_argument(
        "--homography_model_path",
        help="Path to the homography model for perspective correction.",
        type=str,
        default="board_detection/data/models/homomography_hybrid_128_model.pth",
    )
    parser.add_argument(
        "--yolo_model_path",
        help="Path to the YOLO object detector to be used.",
        type=str,
        default="board_detection/data/models/yolo_best.pt",
    )
    return parser.parse_args()


def detect_board(image: Image, model_path: str) -> Image:
    model_yolo = YOLO(model_path)
    class_id = 0
    bbox = board_detection_step(image, model_yolo, class_id, show_results=True)
    bbox = list(map(int, bbox))
    cropped_image = image.crop(bbox)
    return cropped_image


def perspective_correction(img_path: str, model_checkpoint_path: str) -> Image:
    return perspective_correct_image.perspective_correct_image(
        img_path, model_checkpoint_path, model_resolution=128, path_or_img="path"
    )


def extract_hexagons(board_image) -> list:
    checkpoint_path, model_name = (
        "board_segmentation/data/models/sam_vit_b_01ec64.pth",
        "vit_b",
    )
    mask_generator = load_segment_anything(checkpoint_path, model_name)
    np_img = np.array(board_image)
    return extract_single_image_hexagon(np_img, mask_generator, show_plots=True)


def classify_hexagons(hexagon_image_list):
    """
    Classify given hexagons on a board as well as the numbers inside of them.
    @:param hexagon_image_list: A list of PIL images containing segmented hexagons.
    @:return A dictionary containing the hexagon ids, the tile label and the number label
    """

    # reshape to the size expected by the tile detector
    IMG_SIZE = (100, 100, 3)

    args = get_args()

    # load the hexagon detection model
    model = keras.models.load_model(args.hexagon_detector_path)

    # load the label encoder for de-coding the hexagon labels
    with open(args.hexagon_label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    final_dict = {}

    hex_id = 0
    for img in hexagon_image_list:
        pred_hex_label, pred_number_label = predict_image(
            img, model, label_encoder, IMG_SIZE
        )
        final_dict[hex_id] = (pred_hex_label[0], pred_number_label)
        hex_id += 1

    return final_dict


def assemble_board(classified_hexagons, hex_positions=None, image_dimensions=None):
    """
    Assemble the board by determining the adjacency relationships between hexagons.
    This function takes the classified hexagons and creates a structured board representation
    with proper hex grid coordinates and adjacency information based on the standard Catan board layout.

    Args:
        classified_hexagons: Dictionary containing hexagon IDs ('hex_id'), their types ('hex_label'),
                             and number labels ('number_label')
        hex_positions: List of (x, y) coordinates representing the centers of each hexagon.
        image_dimensions: Tuple of (width, height) of the input image.

    Returns:
        Dictionary representing the complete board with adjacency information, grid coordinates,
        and resource distribution statistics.
    """
    board = {"hexagons": {}, "board_layout": {}, "resource_distribution": {}}

    # If we have both hex positions and image dimensions, map detected tiles to standard positions
    if hex_positions is not None and image_dimensions is not None:
        img_width, img_height = image_dimensions
        center_x = img_width / 2
        center_y = img_height / 2
        # Convert hex positions to offsets from center
        offsets = []
        for pos in hex_positions:
            x, y = pos
            offset_x = x - center_x
            offset_y = y - center_y
            offsets.append((offset_x, offset_y))
        # Find the best matching tile for each detected position
        max_std_x = max(abs(x) for x, _ in standard_positions.values())
        max_std_y = max(abs(y) for _, y in standard_positions.values())
        max_offset_x = max(abs(x) for x, _ in offsets)
        max_offset_y = max(abs(y) for _, y in offsets)

        scale_x = max_offset_x / max_std_x if max_std_x > 0 else 1
        scale_y = max_offset_y / max_std_y if max_std_y > 0 else 1

        position_to_id = {}
        used_ids = set()

        for i, (offset_x, offset_y) in enumerate(offsets):
            distances = []
            for tile_id, (std_x, std_y) in standard_positions.items():
                scaled_std_x = std_x * scale_x
                scaled_std_y = std_y * scale_y
                dist = (
                    (offset_x - scaled_std_x) ** 2 + (offset_y - scaled_std_y) ** 2
                ) ** 0.5
                distances.append((dist, tile_id))
            distances.sort()
            for dist, tile_id in distances:
                if tile_id not in used_ids:
                    position_to_id[i] = tile_id
                    used_ids.add(tile_id)
                    break
        # First, populate with known tiles
        for i, (offset_x, offset_y) in enumerate(offsets):
            if i in position_to_id:
                tile_id = position_to_id[i]
                hex_type, hex_number = classified_hexagons[i]
                board["hexagons"][tile_id] = {
                    "type": hex_type,
                    "number": hex_number,
                    "position": standard_positions[tile_id],
                    "grid_coords": standard_positions[tile_id]
                    + (0,),  # Using x,y as grid coords
                    "adjacents": adjacency_map[tile_id],
                }
        # Then make sure all 19 positions are filled, adding unknown for missing ones
        for i in range(19):
            if i not in board["hexagons"]:
                board["hexagons"][i] = {
                    "type": "unknown",
                    "number": None,
                    "position": standard_positions[i],
                    "grid_coords": standard_positions[i] + (0,),
                    "adjacents": adjacency_map[i],
                }
    else:
        # Fallback to standard positions if we don't have enough information
        for i in range(19):
            if i in classified_hexagons:
                hex_type, hex_number = classified_hexagons[i]
                board["hexagons"][i] = {
                    "type": hex_type,
                    "number": hex_number,
                    "position": standard_positions[i],
                    "grid_coords": standard_positions[i] + (0,),
                    "adjacents": adjacency_map[i],
                }
            else:
                board["hexagons"][i] = {
                    "type": "unknown",
                    "number": None,
                    "position": standard_positions[i],
                    "grid_coords": standard_positions[i] + (0,),
                    "adjacents": adjacency_map[i],
                }
    # Create resource distribution counts
    resource_counts = {}
    for hex_data in board["hexagons"].values():
        resource_type = hex_data["type"]
        if resource_type != "unknown":  # Don't count unknown tiles in distribution
            if resource_type in resource_counts:
                resource_counts[resource_type] += 1
            else:
                resource_counts[resource_type] = 1
    # Layout params
    board["resource_distribution"] = resource_counts
    board["board_layout"]["grid"] = None
    board["board_layout"]["dimensions"] = {
        "min_x": min(data["position"][0] for data in board["hexagons"].values()),
        "max_x": max(data["position"][0] for data in board["hexagons"].values()),
        "min_y": min(data["position"][1] for data in board["hexagons"].values()),
        "max_y": max(data["position"][1] for data in board["hexagons"].values()),
    }
    return board


def save_board_to_json(board, output_path):
    """
    Save the assembled board structure to a JSON file.

    Args:
        board: The board dictionary created by assemble_board
        output_path: Path where the JSON file should be saved
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list) or isinstance(obj, tuple):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    json_board = convert_for_json(board)
    try:
        output_path_file = os.path.join(output_path, "board.json")
        with open(output_path_file, "w") as f:
            json.dump(json_board, f, indent=2)
        print(f"Board saved to {output_path_file}")
    except Exception as e:
        print(f"Error saving board to JSON: {e}")


if __name__ == "__main__":
    # CUDA verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    args = get_args()
    IMG_PATH = args.img_path
    HOMOGRAPHY_MODEL_CHECKPOINT_PATH = args.homography_model_path
    YOLO_MODEL_CHECKPOINT_PATH = args.yolo_model_path
    output_path = args.output_path
    # Create a folder for intermediate steps, named after the image name
    image_name = os.path.splitext(os.path.basename(IMG_PATH))[0]
    intermediate_folder = os.path.join(output_path, image_name)
    os.makedirs(intermediate_folder, exist_ok=True)
    # Detect the board
    board_image = perspective_correction(IMG_PATH, HOMOGRAPHY_MODEL_CHECKPOINT_PATH)
    board_image.save(os.path.join(intermediate_folder, "corrected_board.png"))
    # Segment the board
    board_image = detect_board(board_image, YOLO_MODEL_CHECKPOINT_PATH)
    board_image.save(os.path.join(intermediate_folder, "detected_board.png"))
    hexagons, hex_positions = extract_hexagons(board_image)
    # Save the segmented hexagons
    for i, hexagon in enumerate(hexagons):
        hexagon.save(os.path.join(intermediate_folder, f"hexagon_{i}.png"))
    # Classify hexagons and assemble the board
    classified_hexagons_with_numbers = classify_hexagons(hexagons)
    # Assemble the board
    board = assemble_board(
        classified_hexagons_with_numbers,
        hex_positions,
        (board_image.width, board_image.height),
    )
    save_board_to_json(board, intermediate_folder)
    visualize_board(board)

    # classified_hexagons_with_numbers = {
    #     0: ("brick", 8),
    #     1: ("lumber", 10),
    #     2: ("sheep", 4),
    #     3: ("ore", 6),
    #     4: ("wheat", 9),
    #     5: ("ore", 4),
    #     6: ("lumber", 11),
    #     7: ("wheat", 5),
    #     8: ("lumber", 10),
    #     9: ("desert", None),
    #     10: ("brick", 2),
    #     11: ("sheep", 3),
    #     12: ("wheat", 9),
    #     13: ("sheep", 11),
    #     14: ("wheat", 5),
    #     15: ("ore", 12),
    #     16: ("sheep", 6),
    #     17: ("lumber", 8),
    #     18: ("brick", 3),
    # }
