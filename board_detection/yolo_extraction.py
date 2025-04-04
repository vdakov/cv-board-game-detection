import cv2
import numpy as np
import matplotlib.pyplot as plt
from visualization import utils


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
            cv2.putText(
                img,
                f"Class {class_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            print(f"Object found in: {image_path}")
    print(img)
    # Show image only if detections exist
    if len(results) > 0:
        cv2.imshow("Detection", img)
        cv2.waitKey(0)  # Change to 1 ms delay instead of stopping execution


def search_for_object(img_path, model, class_id):
    """Runs YOLO detection on the image."""
    img = cv2.imread(img_path)
    results = model(img)
    return img, results


def board_detection_step(img_path, model, show_results=True, ground_truth_bbox=None):
    """Runs the full detection pipeline and plots results."""
    class_id_to_find = 0
    img, results = search_for_object(img_path, model, class_id_to_find)

    if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
        bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
        img_with_bbox = img.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img_with_bbox_rgb = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)

        if show_results and ground_truth_bbox:
            ground_truth_img = img.copy()
            x1, y1, x2, y2 = map(int, ground_truth_bbox)
            cv2.rectangle(img_with_bbox_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            utils.compare_prediction_ground_truth(
                img.copy(), img_with_bbox_rgb, ground_truth_img
            )
        elif show_results:
            utils.show_before_and_after(
                img.copy(),
                img_with_bbox_rgb,
                title_before="Before Object Detection",
                title_after="After YOLO",
            )

    else:
        print("No boxes detected!")
        return

    return bbox
