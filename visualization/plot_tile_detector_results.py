import matplotlib.pyplot as plt

def plot_loss_accuracy(hist):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot train and validation accuracies
    axes[0].plot(
        hist.history["accuracy"], label="Train accuracy", linestyle="solid", color="b"
    )
    axes[0].plot(
        hist.history["val_accuracy"],
        label="Validation accuracy",
        linestyle="solid",
        color="c",
    )
    axes[0].set_title(f"Accuracies for CNN model on Catan tile set")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Plot train and validation losses
    axes[1].plot(hist.history["loss"], label="Train loss", linestyle="solid", color="b")
    axes[1].plot(
        hist.history["val_loss"], label="Validation loss", linestyle="solid", color="c"
    )
    axes[1].set_title(f"Losses for CNN model on Catan tile set")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_roc(num_classes, fpr, tpr, roc_auc, label_encoder):

    for i in range(num_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f'Class {label_encoder.inverse_transform([i])} (AUC = {roc_auc[i]:.2f})'
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc='lower right')
    plt.show()