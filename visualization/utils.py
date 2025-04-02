import matplotlib.pyplot as plt
import numpy as np

def show_before_and_after(image_before, image_after, title_before="Before", title_after="After"):
    """
    Displays two images side-by-side using matplotlib.

    Args:
        image_before (numpy.ndarray): The image before the transformation.
        image_after (numpy.ndarray): The image after the transformation.
        title_before (str): Title for the "before" image.
        title_after (str): Title for the "after" image.
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with 1 row, 2 columns of subplots

    # Display the "before" image
    axes[0].imshow(image_before)
    axes[0].set_title(title_before)
    axes[0].axis('off')  # Turn off axis labels and ticks

    # Display the "after" image
    axes[1].imshow(image_after)
    axes[1].set_title(title_after)
    axes[1].axis('off')

    plt.tight_layout() # Improves subplot spacing
    plt.show()


def compare_prediction_ground_truth(image, prediction, label, title=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row, 2 columns of subplots

    # Display the "before" image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')  # Turn off axis labels and ticks

    # Display the "after" image
    axes[1].imshow(prediction)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    # Display the "after" image
    axes[2].imshow(label)
    axes[2].set_title("Label")
    axes[2].axis('off')

    if title:
        plt.title(title)

    plt.tight_layout() # Improves subplot spacing
    plt.show()

