from keras import layers, models, callbacks, Sequential, optimizers, utils, initializers
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
import json
import torch
import ast
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import os
import random
from visualization.plot_tile_detector_results import plot_roc, plot_loss_accuracy


def to_tensor(tensor_str):
    # Convert string to a list using ast.literal_eval
    tensor_list = ast.literal_eval(tensor_str)
    return torch.tensor(tensor_list)


def model_eval(model, ds_test, X_test, y_test, num_classes):

    # First get the test loss and accuracy
    ds_test = ds_test.batch(BATCH_SIZE)
    loss, acc = model.evaluate(ds_test, verbose=2)

    # Then get the prediction to compute other metrics
    X_test_np = X_test.numpy()
    y_pred = model.predict(X_test_np)

    y_test_np = utils.to_categorical(y_test, num_classes=num_classes)
    y_test_np = y_test_np.numpy()

    # Compute the roc curves and the AUC
    fpr = {}
    tpr = {}
    roc_auc = {}

    final_auc = roc_auc_score(y_test_np, y_pred, average="macro", multi_class="ovr")

    for i in range(y_test_np.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test_np[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    map_score = average_precision_score(y_test_np, y_pred, average="macro")

    return loss, acc, fpr, tpr, roc_auc, final_auc, map_score


def model_predict(model, sample, label_encoder, img_size):
    # Read the image
    img = Image.open(sample).convert("RGB")
    img_np = ToTensor()(img)
    img_np = torch.nn.functional.interpolate(img_np.unsqueeze(0), size=img_size[:2])
    img_np = img_np.permute(0, 2, 3, 1)
    img_np = img_np.numpy()

    pred = model.predict(img_np)

    pred_label = label_encoder.inverse_transform([np.argmax(pred)])

    print(f"Image at path: {sample} is a {pred_label} tile")


def model_training(model, train_set, valid_set, epochs, save_path):
    # Stop when the loss does not improve significantly over 3 epochs
    callback = callbacks.EarlyStopping(monitor="loss", min_delta=0.01, patience=3)

    # Keras expects a batched dataset
    batched_train = train_set.batch(BATCH_SIZE)
    batched_valid = valid_set.batch(BATCH_SIZE)

    # Fit the model to the training data
    hist = model.fit(
        x=batched_train,
        validation_data=batched_valid,
        epochs=epochs,
        callbacks=[callback],
    )

    # Plot training loss and accuracy
    plot_loss_accuracy(hist, save_path)

    return model


def data_augmentation(input_size):
    # Data augmentation component
    return Sequential(
        [
            layers.Input(input_size),
            layers.RandomFlip("horizontal"),
            layers.RandomFlip("vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.2),
        ]
    )


def build_dataset(dataset_path, valid_split, test_split):

    print(f"Compiling dataset at path: {dataset_path}")

    with open(dataset_path, "rb") as f:
        (X, y) = pickle.load(f)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    valid_size = int(valid_split * len(X))
    test_size = int(test_split * len(X))
    train_size = len(X) - valid_size - test_size

    final_indices = [
        indices[:train_size],
        indices[train_size : train_size + valid_size],
        indices[train_size + valid_size :],
    ]

    # Define training set
    X_train = tf.gather(X, final_indices[0])
    y_train = tf.gather(y, final_indices[0])

    # Pass the training set only through the data augmentation pipeline
    augmentation = data_augmentation(X_train.shape[1:])
    X_train = augmentation(X_train)

    # Define validation set
    X_valid = tf.gather(X, final_indices[1])
    y_valid = tf.gather(y, final_indices[1])

    # Define test set
    X_test = tf.gather(X, final_indices[2])
    y_test = tf.gather(y, final_indices[2])

    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]


def build_cnn(input_shape, seed):

    # Initialize all weights with the same seed to ensure reproducibility
    initializer = initializers.RandomNormal(seed=seed)

    # Define CNN model
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(
                25,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="valid",
                activation="relu",
                input_shape=input_shape,
                kernel_initializer=initializer,
            ),
            layers.Dropout(0.15),
            layers.MaxPool2D(pool_size=(1, 1), padding="valid"),
            layers.Flatten(),
            layers.Dense(80, activation="relu", kernel_initializer=initializer),
            layers.Dropout(0.15),
            layers.Dense(
                NUM_CLASSES, activation="softmax", kernel_initializer=initializer
            ),
        ]
    )

    # Define optimizer
    optimizer = optimizers.Adam(learning_rate=5e-4, use_ema=True)

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer=optimizer,
    )

    return model


if __name__ == "__main__":

    # Fix error where multiple libiomp5md.dll files are present
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # Seed everything to ensure reproducibility
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    ##### PARAMETER DEFINITION #####

    # Downsample all images for faster training
    IMG_SIZE = (100, 100, 3)
    BATCH_SIZE = 32
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 100  # the maximum number of epochs used to train the model
    validation_split = 0.2
    test_split = 0.1
    path_to_predict = "data/input/test1.png"
    model_save_path = "data/models/tile_detector_hexagons.keras"
    model_result_save_path = "data/output/tile_detector_hexagon_test_results.txt"
    dataset_path = "data/output/compiled_dataset/synthetic_dataset_hexagons.pkl"
    label_encoder_path = (
        "data/output/compiled_dataset/label_encoder/label_encoder_hexagons.pkl"
    )
    train_plot_save_path = "data/output/hex_detector_training_plot.png"
    roc_curve_save_path = "data/output/hex_detector_roc_curve.png"

    ##### DATASET PRE-PROCESSING #####

    final_sets = build_dataset(dataset_path, validation_split, test_split)

    X_train, y_train = final_sets[0]
    train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))

    X_valid, y_valid = final_sets[1]
    validation_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    X_test, y_test = final_sets[2]
    test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    ##### TRAINING THE MODEL #####

    # First build the model
    model = build_cnn(IMG_SIZE, seed)

    # Then train it on the input data
    # This will also plot the train and validation accuracies
    model = model_training(
        model, train_set, validation_set, epochs, train_plot_save_path
    )

    model.save(model_save_path)
    print(f"Trained model saved at: {model_save_path}")

    ##### TESTING THE MODEL #####

    # Test the model on the test set
    loss, acc, fpr, tpr, roc_auc, final_auc, map_score = model_eval(
        model, test_set, X_test, y_test, NUM_CLASSES
    )

    model_result_dict = {
        "test_loss": loss,
        "test_acc": acc,
        "test_map_score": map_score,
        "test_auc": final_auc,
    }

    with open(model_result_save_path, "w") as file:
        json.dump(model_result_dict, file)

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Plot the ROC curve
    plot_roc(NUM_CLASSES, fpr, tpr, roc_auc, label_encoder, roc_curve_save_path)

    # Predict a single sample that is different from the test set
    model_predict(model, path_to_predict, label_encoder, IMG_SIZE[:2])
