from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


data_dir = "data/sample/perspective_distorted_boards_sample/images"
model_path = "runs/detect/train10/weights/best.pt"  # Replace with your trained model
class_id_to_find = 0  # Change this to the class ID of the object you want to search for

model = YOLO(model_path)

# Load reference image and extract center descriptor
ref_img = cv2.imread("data/sample/mined_synthetic_boards_sample/canvas_image_2.png", cv2.IMREAD_GRAYSCALE)
h, w = ref_img.shape
sift = cv2.SIFT_create()
center_keypoint = [cv2.KeyPoint(x=w//2, y=h//2, size=20)]
_, ref_descriptor = sift.compute(ref_img, center_keypoint)


# def search_for_object(img_path, model, class_id, debug=False):
#     img = cv2.imread(img_path)
#     results = model(img)
#     if debug:
#         show_bbox(results, cv2.imread(img_path), img_path, class_id)
#     return img, results


# def show_bbox(results, img, image_path, class_id):
#     for result in results:
#         if not hasattr(result, "boxes"):
#             continue  # Skip if no boxes are detected
#         print(img)
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(img, f"Class {class_id}", (x1, y1 - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#             print(f"Object found in: {image_path}")
#     print(img)
#     # Show image only if detections exist
#     if len(results) > 0:
#         cv2.imshow("Detection", img)
#         cv2.waitKey(0)  # Change to 1 ms delay instead of stopping execution



# def detect_and_correct_perspective(img, bbox):
#     """Corrects perspective using a single SIFT descriptor from the center."""
#     x1, y1, x2, y2 = map(int, bbox)
    
#     # Crop detected board
#     cropped_board = img[y1:y2, x1:x2]

#     # Convert to grayscale for SIFT
#     gray_cropped = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2GRAY)

#     # Detect the center keypoint in the cropped image
#     h_crop, w_crop = gray_cropped.shape
#     test_keypoint = [cv2.KeyPoint(x=w_crop//2, y=h_crop//2, size=20)]
#     _, test_descriptor = sift.compute(gray_cropped, test_keypoint)

#     # Ensure a descriptor was found
#     if test_descriptor is None:
#         print("No keypoint found in cropped board.")
#         return img

#     # Match descriptors using Brute Force Matcher
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(ref_descriptor, test_descriptor)

#     if len(matches) == 0:
#         print("No match found.")
#         return img

#     # Get matched keypoint locations
#     ref_pt = np.float32([center_keypoint[0].pt])
#     test_pt = np.float32([test_keypoint[0].pt])

#     # Compute affine transform
#     M = cv2.estimateAffinePartial2D(test_pt, ref_pt)[0]

#     # Warp the image using the affine transformation
#     corrected_board = cv2.warpAffine(cropped_board, M, (w, h))

#     # Show the corrected image
#     cv2.imshow("Corrected Board", corrected_board)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return corrected_board

# def board_detection_step(img_path, model_path):
#     model = YOLO(model_path)
#     img, results = search_for_object(img_path, model, 0, debug=True)
#     if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
#         bbox = results[0].boxes[0].xyxy[0]  # Get first bbox
#         img = detect_and_correct_perspective(img, bbox)

#     return img

# board_detection_step("data/full/perspective_distorted_boards/val/images/canvas_image_3.png", model_path)

## Try a set of permutations itself 


def search_for_object(img_path, model, class_id):
    """Runs YOLO detection on the image."""
    img = cv2.imread(img_path)
    results = model(img)
    return img, results

# Detect keypoints and descriptors in reference image
kp_ref, des_ref = sift.detectAndCompute(ref_img, None)


def correct_rotation(img, bbox, ref_angle):
    """Corrects rotation using a single SIFT keypoint orientation."""
    x1, y1, x2, y2 = map(int, bbox)
    cropped_board = img[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray_cropped = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(gray_cropped, None)

    if not kp:
        print("No keypoints found.")
        return cropped_board

    # Use the first keypoint's angle (assuming it is stable)
    detected_angle = kp[0].angle
    angle_diff = ref_angle - detected_angle

    # Compute rotation matrix
    h, w = cropped_board.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_diff, 1)

    # Rotate image
    rotated_board = cv2.warpAffine(cropped_board, M, (w, h))

    return rotated_board

def visualize_sift_keypoint(img, keypoint, title="SIFT Keypoint"):
    """Draw and visualize a single SIFT keypoint on the image."""
    img_with_keypoint = cv2.drawKeypoints(img, keypoint, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.imshow(cv2.cvtColor(img_with_keypoint, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


    def detect_and_correct_perspective(img, bbox):
        """Detects board angle using SIFT and corrects rotation."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop detected board
        cropped_board = img[y1:y2, x1:x2]
        gray_cropped = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2GRAY)
        kp_crop, _ = sift.detectAndCompute(gray_cropped, None)

        if not kp_crop:
            print("No keypoints found in detected board.")
            return img, cropped_board, None

        detected_angle = kp_crop[0].angle  # Take first keypoint angle
        
        ref_angle = kp_ref[0].angle  # Take first keypoint angle
        print(detected_angle, ref_angle)

        print(f"Reference Angle: {ref_angle}, Detected Angle: {detected_angle}")

        # Visualize keypoints
        visualize_sift_keypoint(ref_img, [kp_ref[0]], "Reference Keypoint")
        visualize_sift_keypoint(cropped_board, [kp_crop[0]], "Detected Keypoint")


        rotated_board = correct_rotation(cropped_board, bbox, ref_angle)

        return img, cropped_board, rotated_board



def board_detection_step(img_path, model):
    """Runs the full detection pipeline and plots results."""
    img, results = search_for_object(img_path, model, class_id_to_find)

    if results and hasattr(results[0], "boxes") and len(results[0].boxes) > 0:
        bbox = results[0].boxes[0].xyxy[0].cpu().numpy()
        img_with_bbox = img.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

        img, cropped_board, corrected_board = detect_and_correct_perspective(img, bbox)

        # Convert images to RGB for Matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_with_bbox_rgb = cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped_board, cv2.COLOR_BGR2RGB)
        corrected_rgb = cv2.cvtColor(corrected_board, cv2.COLOR_BGR2RGB) if corrected_board is not None else None

        # Plot all images
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(img_with_bbox_rgb)
        axes[1].set_title("Detected Board")
        axes[1].axis("off")

        axes[2].imshow(cropped_rgb)
        axes[2].set_title("Cropped Board")
        axes[2].axis("off")

        if corrected_rgb is not None:
            axes[3].imshow(corrected_rgb)
            axes[3].set_title("Corrected Perspective")
        else:
            axes[3].imshow(np.zeros_like(cropped_rgb))  # Empty placeholder if correction fails
            axes[3].set_title("Correction Failed")
        axes[3].axis("off")

        plt.tight_layout()
        plt.show()

    return img


# Run detection & plot results
board_detection_step("data/full/perspective_distorted_boards/train/images/canvas_image_0.png", model)