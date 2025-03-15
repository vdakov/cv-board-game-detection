from ultralytics import YOLO
import cv2
import os
import glob


data_dir = "data/full/perspective_distorted_boards/images"
model_path = "runs/detect/train10/weights/best.pt"  # Replace with your trained model
class_id_to_find = 0  # Change this to the class ID of the object you want to search for


def search_for_object(img_path, model, class_id, debug=False):
    img = cv2.imread(img_path)
    results = model(img)
    if debug:
        show_bbox(results, cv2.imread(img_path), img_path, class_id)
    return img, results


def show_bbox(results, img, image_path, class_id):
    for result in results:
        if not hasattr(result, "boxes"):
            continue  # Skip if no boxes are detected
        print(img)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Object found in: {image_path}")
    print(img)
    # Show image only if detections exist
    if len(results) > 0:
        cv2.imshow("Detection", img)
        cv2.waitKey(0)  # Change to 1 ms delay instead of stopping execution




def detect_and_correct_perspective(img, bbox):
    pass

def board_detection_step(img_path, model_path):
    model = YOLO(model_path)
    img, results = search_for_object(img_path, model, 0, debug=True)
    if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
        bbox = results[0].boxes[0].xyxy[0]  # Get first bbox
        img = detect_and_correct_perspective(img, bbox)

    return img

board_detection_step("data/full/perspective_distorted_boards/val/images/canvas_image_3.png", model_path)


