import argparse
import keras
import pickle
from board_piece_classification.hexagon_prediction import predict_image
from PIL import Image
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
        default="board_piece_classification/data/models/tile_detector_hexagons2.keras",
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
        default="data/board.json",
    )
    return parser.parse_args()


def detect_and_perspective_correct_board(IMG_PATH):
    pass


def extract_hexagons(board_image):
    pass


def classifiy_hexagons(hexagon_image_folder):
    """ "
    Classify given hexagons on a board as well as the numbers inside of them.
    @:param image_folder: The folder containing images of hexagons to be classified.
    @:return A dictionary containing the hexagon ids, the tile label and the number label
    """

    # reshape to the size expected by the tile detector
    IMG_SIZE = (243, 256, 3)

    args = get_args()

    # load the hexagon detection model
    model = keras.models.load_model(args.hexagon_detector_path)

    # load the label encoder for de-coding the hexagon labels
    with open(args.hexagon_label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    final_dict = {"hex_id": [], "hex_label": [], "number_label": []}

    hex_id = 1
    for img in os.listdir(hexagon_image_folder):
        img_path = f"{hexagon_image_folder}/{img}"

        pred_hex_label, pred_number_label = predict_image(
            img_path, model, label_encoder, IMG_SIZE
        )

        final_dict["hex_id"].append(hex_id)
        final_dict["hex_label"].append(pred_hex_label)
        final_dict["number_label"].append(pred_number_label)

        hex_id += 1

    return final_dict


def assemble_board(classified_hexagons, hex_positions=None):
    """
    Assemble the board by determining the adjacency relationships between hexagons.
    This function takes the classified hexagons and creates a structured board representation
    with proper hex grid coordinates and adjacency information based on the standard Catan board layout.

    Args:
        classified_hexagons: Dictionary containing hexagon IDs ('hex_id'), their types ('hex_label'),
                             and number labels ('number_label')
        hex_positions: List of (x, y) coordinates representing the centers of each hexagon.
                       If None, will use a predefined layout for standard Catan boards.

    Returns:
        Dictionary representing the complete board with adjacency information, grid coordinates,
        and resource distribution statistics.
    """
    # Guard
    if not classified_hexagons or "hex_id" not in classified_hexagons:
        raise ValueError("classified_hexagons must contain a 'hex_id' key")
    num_hexes = len(classified_hexagons["hex_id"])
    if num_hexes != 19:
        print(f"Warning: Expected 19 hexagons, not {num_hexes}")

    board = {"hexagons": {}, "board_layout": {}, "resource_distribution": {}}
    # Create hexagons with the defined layout
    for i in range(min(num_hexes, 19)):
        hex_id = classified_hexagons["hex_id"][i]
        hex_type = classified_hexagons["hex_label"][i]
        hex_number = classified_hexagons["number_label"][i]
        x, y = standard_positions[hex_id]
        board["hexagons"][hex_id] = {
            "id": hex_id,
            "type": hex_type,
            "number": hex_number,
            "position": (x, y),
            "grid_coords": (x, y, 0),  # Using x,y as grid coords for simplicity
            "adjacents": adjacency_map[hex_id],  # Use predefined adjacencies
        }
    # Create resource distribution counts
    resource_counts = {}
    for hex_data in board["hexagons"].values():
        resource_type = hex_data["type"]
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
    # Create the output directory if it doesn't exist
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

    # Convert the board to a JSON-serializable format
    json_board = convert_for_json(board)

    # Save to file
    try:
        with open(output_path, "w") as f:
            json.dump(json_board, f, indent=2)
        print(f"Board saved to {output_path}")
    except Exception as e:
        print(f"Error saving board to JSON: {e}")


if __name__ == "__main__":
    args = get_args()
    IMG_PATH = args.img_path
    output_path = args.output_path

    # Process the board image
    board_image = detect_and_perspective_correct_board(IMG_PATH)
    hexagons = extract_hexagons(board_image)

    # Extract the centers of the hexagons for position information
    hex_positions = []
    try:
        # Try to extract positions from the hexagon images
        for hex_img_path in sorted(os.listdir(hexagons)):
            if hex_img_path.endswith(".png") or hex_img_path.endswith(".jpg"):
                img = Image.open(os.path.join(hexagons, hex_img_path))
                # Get the center of the hexagon image in its original context
                # This assumes extract_hexagons stores the original position somehow
                # Ideally, extract_hexagons would return both images and their positions
                width, height = img.size
                center = (width // 2, height // 2)
                hex_positions.append(center)

        if not hex_positions:
            print(
                "Warning: Could not extract hex positions from images. Will use default positions."
            )
            hex_positions = None
    except Exception as e:
        print(f"Error extracting hex positions: {e}")
        print("Will use default positions instead.")
        hex_positions = None

    # Classify hexagons and assemble the board
    classified_hexagons_with_numbers = classifiy_hexagons(hexagons)

    # classified_hexagons_with_numbers = {
    #     "hex_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    #     "hex_label": [
    #         "wheat",
    #         "sheep",
    #         "lumber",
    #         "wheat",
    #         "desert",
    #         "wheat",
    #         "lumber",
    #         "wheat",
    #         "brick",
    #         "brick",
    #         "sheep",
    #         "ore",
    #         "sheep",
    #         "lumber",
    #         "sheep",
    #         "ore",
    #         "brick",
    #         "ore",
    #         "lumber",
    #     ],
    #     "number_label": [12, 3, 2, 10, 0, 5, 6, 11, 5, 11, 8, 10, 4, 8, 9, 3, 9, 6, 4],
    # }

    board = assemble_board(classified_hexagons_with_numbers)
    save_board_to_json(board, output_path)
    visualize_board(board)
