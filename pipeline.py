import argparse
import keras
import pickle
from board_piece_classification.hexagon_prediction import predict_hexagons, preprocess_image
from PIL import Image
import pytesseract
import os
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--img_path", help="Input the path to your image here.", type=str, default="data/sample/mined_synthetic_boards_sample/canvas_image_0.png")
    parser.add_argument("--hexagon_detector_path", help="Input the path to the model that detects hexagons here.", type=str, default="board_piece_classification/model/tile_detector_hexagons2.keras")
    parser.add_argument("--hexagon_label_encoder_path", help="Input the path to the label encoder associated with the hexagon detector here.", type=str, default="data/full/compiled_dataset/label_encoder/label_encoder_hexagons.pkl")
    return args

def detect_and_perspective_correct_board(IMG_PATH):
    pass

def extract_hexagons(board_image):
    pass

def classifiy_hexagon_types(hexagon_image_folder):
    """"
    Classify the hexagons as well as th numbers inside of them.
    @:param hexagon_images: The hexagon features
    @:return A dictionary containing the hexagon ids, the tile label and the number label
    """

    # reshape to the size expected by the tile detector
    IMG_SIZE = (243, 256, 3)

    args = get_args()

    # load the hexagon detection model
    model = keras.models.load_model(args.hexagon_detector_path)

    # load the label encoder for de-coding the hexagon labels
    with open(args.hexagon_label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    final_dict = {
        "hex_id": [],
        "hex_label": [],
        "number_label": []
    }

    hex_id = 1
    for img in os.listdir(hexagon_image_folder):
        img_path = f'{hexagon_image_folder}/{img}'

        final_dict["hex_id"].append(hex_id)

        pred_hex_label = predict_hexagons(model, img_path, label_encoder, IMG_SIZE[:2])
        final_dict["hex_label"].append(pred_hex_label)

        img1 = np.array(preprocess_image(Image.open(img_path), 3.5, 85))
        text = pytesseract.image_to_string(img1, config='--psm 3')
        final_dict["number_label"].append(text)

    return final_dict


def assign_numbers_to_hexagons(classified_hexagons):
    pass 

def assemble_board(classified_hexagons_with_numbers):
    pass

def visualize_board(board):
    pass

if __name__ == "__main__":
    args = get_args()
    IMG_PATH = args.img_path
    board_image = detect_and_perspective_correct_board(IMG_PATH)
    hexagons = extract_hexagons(board_image)
    classified_hexagons = classifiy_hexagon_types(hexagons)
    classified_hexagons_with_numbers = assign_numbers_to_hexagons(classified_hexagons)
    board = assemble_board(classified_hexagons_with_numbers)
    visualize_board(board)


            



   
