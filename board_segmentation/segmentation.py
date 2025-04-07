import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from board_segmentation.mask_clustering import cluster_masks


def filter_for_hexagons(original, anns, show_plots=False):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=lambda x: x["area"], reverse=True)

    if show_plots:
        _, ax = plt.subplots(1, 3, figsize=(15, 5))

    sorted_anns = cluster_masks(sorted_anns)

    H, W = sorted_anns[0]["segmentation"].shape
    img = np.zeros((H, W, 4))  # Start with black image (RGBA)

    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m, :] = color_mask  #

    cluster_img = np.zeros((H, W, 4))

    unique_clusters = set(ann["cluster"] for ann in sorted_anns if ann["cluster"] != -1)
    print("Estimated number of clusters: %d" % len(unique_clusters))
    print(
        "Estimated number of noise points: %d"
        % len(set(ann["cluster"] for ann in sorted_anns if ann["cluster"] == -1))
    )
    cluster_colors = {
        c: np.concatenate([np.random.random(3), [0.7]]) for c in unique_clusters
    }

    for ann in sorted_anns:
        cluster_label = ann["cluster"]
        if cluster_label != -1:  # Ignore outliers
            m = ann["segmentation"]
            cluster_img[m, :] = cluster_colors[cluster_label]

    if show_plots:
        ax[0].imshow(original)
        ax[1].imshow(img)
        ax[2].imshow(cluster_img)
        plt.axis("off")
        plt.show()

    del sorted_anns

    return cluster_img


def segment_all(mask_generator, img):
    # Generate masks and release image to avoid memory overload
    masks = mask_generator.generate(img)
    del img
    torch.cuda.empty_cache()  # Clear GPU memory

    return masks


# def calculate_angle(p1, p2, p3):
#     (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3

#     b = np.abs(x1 - x2)
#     h_b = np.abs(y2 - y1)
#     d = np.abs(x2 - x3)
#     h_d = np.abs(y2 - y3)

#     theta_1 = np.arctan(h_b / b)
#     theta_2 = np.arctan(h_d / d)

#     assert theta_1 + theta_2 < np.pi / 2

#     theta = (np.pi / 2) - theta_1 - theta_2

#     return theta


def filter_close_points(points, min_dist_ratio, image_shape):
    min_dist = min_dist_ratio * min(image_shape[:2])
    filtered = [points[0]]

    for pt in points[1:]:
        if np.linalg.norm(pt - filtered[-1]) >= min_dist:
            filtered.append(pt)

    if np.linalg.norm(filtered[-1] - filtered[0]) < min_dist and len(filtered) > 1:
        filtered.pop()

    return np.array(filtered)


def calculate_angle(p1, p2, p3):
    # Create vectors from p2 to p1 and p2 to p3
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    # Compute the dot product and norms
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0  # or raise an exception, as appropriate

    # Compute the cosine of the angle and then the angle itself
    cos_angle = np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    print("Candidate candle", angle)

    return angle


def extract_hexagon_contours(cluster_img, show_plots=False):
    cluster_img = (cluster_img * 255).astype(np.uint8)
    cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)
    _, cluster_img = cv2.threshold(cluster_img, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cluster_img = cv2.morphologyEx(cluster_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cluster_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if show_plots:
        result = cv2.cvtColor(cluster_img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 5))
        plt.imshow(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        )  # Convert for matplotlib display
        plt.axis("off")
        plt.show()
    # Reference area threshold: ignore if too small
    image_area = cluster_img.shape[0] * cluster_img.shape[1]
    min_area = image_area / 50.0  # anything smaller than 1/50th of image is skipped

    hexagons = []
    hexagon_areas = []
    tolerance = 0.25

    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # First pass: collect hexagons and their areas
    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corners = approx.reshape(-1, 2)
        corners = filter_close_points(corners, 0.05, cluster_img.shape)

        # print("Candidate", len(corners))

        if len(corners) == 6:

            angles = [
                calculate_angle(
                    corners[i - 1 % len(corners)],
                    corners[i],
                    corners[(i + 1) % len(corners)],
                )
                for i in range(len(corners))
            ]
            total_angle = sum(angles)
            # print("Total angle", math.degrees(total_angle))
            # print("Angles", list(map(math.degrees, angles)))

            if (705 * np.pi / 180) <= total_angle <= (735 * np.pi / 180):  # 720° ± 15

                area = cv2.contourArea(approx)
                if area > min_area:
                    hexagons.append(approx)
                    hexagon_areas.append(area)

    if len(hexagon_areas) == 0:
        return []

    average_area = np.mean(hexagon_areas)
    tolerance = 0.25  # ±25% tolerance

    filtered_hexagons = [
        hex
        for hex, area in zip(hexagons, hexagon_areas)
        if (1 - tolerance) * average_area <= area <= (1 + tolerance) * average_area
    ]

    return filtered_hexagons
