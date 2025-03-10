from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

def get_ontology():
    """
    :return: the ontology used to segment the board into hexagonal tiles and number plates
    """
    ontology = {
        'hexagon': 'hexagonal area inside yellow borders',
        'circle': 'black border. white area. number inside.'
    }
    return ontology

def get_annotated_images(input_data_path, output_data_path):
    base_model = GroundedSAM(ontology=CaptionOntology(get_ontology()))

    pth = "../catan_data/mined_synthetic_boards_sample/canvas_image_0.png"

    results = base_model.predict(pth)

    plot(
        image=cv2.imread(pth),
        classes=base_model.ontology.classes(),
        detections=results
    )

input_data_path = ''
output_data_path = ''

get_annotated_images(input_data_path, output_data_path)