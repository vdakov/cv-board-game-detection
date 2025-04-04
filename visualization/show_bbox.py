import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_bounding_box(bbox, ax=None, edgecolor="red", linewidth=2, label=None):
    """
    Draws a bounding box on a given matplotlib Axes.

    Parameters:
        bbox (tuple or list): The bounding box as (x, y, width, height).
        ax (matplotlib.axes.Axes, optional): The axes to draw on. If None, a new figure and axes are created.
        edgecolor (str): Color of the bounding box edge.
        linewidth (int or float): Width of the bounding box edge.
        label (str, optional): Label for the bounding box (optional, used in legend).

    Returns:
        ax (matplotlib.axes.Axes): The axes with the bounding box drawn.
    """
    polygon = plt.Polygon(
        bbox, edgecolor=edgecolor, fill=False, linewidth=linewidth, label=label
    )
    ax.add_patch(polygon)


def show_two_bounding_boxes(
    bbox1, bbox2, image1=None, image2=None, edgecolor1="red", edgecolor2="blue"
):
    """
    Displays two bounding boxes in side-by-side subplots.

    Parameters:
        bbox1 (tuple or list): The bounding box for the first subplot as (x, y, width, height).
        bbox2 (tuple or list): The bounding box for the second subplot as (x, y, width, height).
        image1 (array-like, optional): The first image to display in subplot 1 (if provided).
        image2 (array-like, optional): The second image to display in subplot 2 (if provided).
        edgecolor1 (str): Edge color for the first bounding box.
        edgecolor2 (str): Edge color for the second bounding box.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the first image if provided, otherwise configure the axis
    if image1 is not None:
        ax1.imshow(image1)
    else:
        # Set default limits if no image is provided
        ax1.set_xlim(0, 100)
        ax1.set_ylim(100, 0)  # Invert y-axis for image-like display

    # Display the second image if provided, otherwise configure the axis
    if image2 is not None:
        ax2.imshow(image2)
    else:
        ax2.set_xlim(0, 100)
        ax2.set_ylim(100, 0)

    # Draw the bounding boxes on each subplot
    show_bounding_box(bbox1, ax=ax1, edgecolor=edgecolor1, label="Box 1")
    show_bounding_box(bbox2, ax=ax2, edgecolor=edgecolor2, label="Box 2")

    plt.tight_layout()
    plt.show()
