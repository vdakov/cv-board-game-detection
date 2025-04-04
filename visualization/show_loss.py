import matplotlib.pyplot as plt
import numpy as np


def show_train_test_loss(
    training_losses, test_loss, num_epochs, title="Training and Test Loss Over Epochs"
):
    """
    Generates a visually appealing plot of training loss and a horizontal line for test loss.

    Args:
        training_losses (list): List of training losses per epoch.
        test_loss (float): Single test loss value.
        num_epochs (int): Total number of epochs.
        title (str): The title of the plot.
    """

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 6), dpi=100)

    plt.plot(
        epochs,
        training_losses,
        marker="o",
        linestyle="-",
        color="#3498db",
        label="Training Loss",
        linewidth=2,
        markersize=6,
    )
    plt.axhline(y=test_loss, color="#e74c3c", linestyle="--", label="Test Loss")

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    plt.xticks(epochs)
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.legend(fontsize=12)
    plt.tight_layout()

    ax = plt.gca()
    ax.set_facecolor("#f4f4f4")

    for spine in ax.spines.values():
        spine.set_edgecolor("gray")
        spine.set_linewidth(0.5)

    plt.show()
