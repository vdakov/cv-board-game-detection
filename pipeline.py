import argparse

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    parser.add_argument("--img_path", help="Input the path to your image here.", type=str, default="data/sample/mined_synthetic_boards_sample/canvas_image_0.png")
    return args

def detect_and_perspective_correct_board(IMG_PATH):
    pass

def extract_hexagons(board_image):
    pass

def classifiy_hexagon_types(hexagons):
    pass

def assign_numbers_to_hexagons(classified_hexagons)
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


            



   
