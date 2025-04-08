import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def calculate_r_squared(
    predicted_matrices, ground_truth_matrices, title="Model Performance"
):
    """
    Calculates the R-squared score comparing
    predicted values against ground truth values.

    Args:
        predicted_matrix (numpy.ndarray): Matrix of predicted values.
        ground_truth_matrix (numpy.ndarray): Matrix of ground truth values.
        title (str, optional): Title of the plot. Defaults to "Model Performance".

    Returns:
        float: The calculated R-squared score.
    """

    all_predicted_values = np.concatenate(
        [matrix.flatten() for matrix in predicted_matrices]
    )
    all_ground_truth_values = np.concatenate(
        [matrix.flatten() for matrix in ground_truth_matrices]
    )

    r_squared = r2_score(all_ground_truth_values, all_predicted_values)

    return r_squared


def frobenius_norm(matrix1, matrix2):
    """Calculates the Frobenius norm of the difference between two matrices. A measure of the distance between them"""
    return np.linalg.norm(matrix1 - matrix2, "fro")


def average_corner_error(matrix_pred, matrix_gt, bbox):
    """Calculates the Average Corner Error between two homography matrices. A random bounding box is generated for the image, and it is measured how much the corners differ.
    Source: https://arxiv.org/pdf/1606.03798 Deep Image Homography Estimation"""

    inverse_matrix_pred = np.linalg.inv(matrix_pred)
    inverse_matrix_gt = np.linalg.inv(matrix_gt)

    prediction_corners = inverse_matrix_pred @ bbox
    gt_corners = inverse_matrix_gt @ bbox

    prediction_corners = prediction_corners / prediction_corners[2]
    gt_corners = gt_corners / gt_corners[2]

    return np.mean(np.linalg.norm(gt_corners - prediction_corners))


def calculate_mean_frobenius_norm(predicted_homographies, ground_truth_homographies):
    """
    Calculates the mean Frobenius norm for an entire dataset.

    Args:
        predicted_homographies (list or numpy.ndarray): A list or array of predicted homography matrices.
        ground_truth_homographies (list or numpy.ndarray): A list or array of ground truth homography matrices.

    Returns:
        float: The mean Frobenius norm over the dataset.
    """
    if len(predicted_homographies) != len(ground_truth_homographies):
        raise ValueError(
            "The number of predicted and ground truth homographies must be the same."
        )

    frobenius_norms = [
        frobenius_norm(pred_H, gt_H)
        for pred_H, gt_H in zip(predicted_homographies, ground_truth_homographies)
    ]
    return np.mean(frobenius_norms)


def calculate_mean_corner_error(
    predicted_homographies, ground_truth_homographies, image_shape=(128, 128)
):
    """
    Calculates the mean Average Corner Error for an entire dataset.

    Args:
        predicted_homographies (list or numpy.ndarray): A list or array of predicted homography matrices.
        ground_truth_homographies (list or numpy.ndarray): A list or array of ground truth homography matrices.
        image_shape (tuple): The (height, width) of the images in your dataset.
        num_corners (int): The number of corners to use for ACE calculation (typically 4).

    Returns:
        float: The mean Average Corner Error over the dataset (in pixels).
    """
    if len(predicted_homographies) != len(ground_truth_homographies):
        raise ValueError(
            "The number of predicted and ground truth homographies must be the same."
        )

    corner_errors = []
    height, width = image_shape
    corners = np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
    ).T  # Shape (3, 4)

    for pred_H, gt_H in zip(predicted_homographies, ground_truth_homographies):
        corner_errors.append(average_corner_error(pred_H, gt_H, corners))

    return np.mean(corner_errors)
