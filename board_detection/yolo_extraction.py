import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from visualization import utils


def search_for_object(img, model, class_id, debug=False):
    results = model(img)
    if debug:
        show_bbox(results, img, class_id)
    return img, results


def show_bbox(results, img, class_id):
    img = np.array(img)

    for result in results:
        if not hasattr(result, "boxes"):
            continue  # Skip if no boxes are detected
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
    # Show image only if detections exist
    if len(results) > 0:
        cv2.imshow("Detection", img)
        cv2.waitKey(0)  # Change to 1 ms delay instead of stopping execution
        cv2.destroyAllWindows()


def board_detection_step(input, model, class_id, show_results=True):
    """Runs the full detection pipeline and plots results."""


    results = search_for_object(input, model, class_id)


    if results and hasattr(results[1][0], "boxes") and len(results[1][0].boxes) > 0:
        bbox = results[1][0].boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)

        img_np = np.array(input.copy())  # Convert PIL to NumPy
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_results:
            utils.show_before_and_after(
                np.array(input),
                img_np,
                title_before="Before Detection",
                title_after="After YOLO"
            )

        return bbox
    else:
        print("No boxes detected!")
        return None


