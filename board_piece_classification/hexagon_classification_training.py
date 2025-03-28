import keras
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import torch
import ast
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
import os

def plot_hist(hist):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot train and validation accuracies
    axes[0].plot(hist.history["accuracy"], label="Train accuracy", linestyle='solid', color='b')
    axes[0].plot(hist.history["val_accuracy"], label="Validation accuracy", linestyle='dashed', color='c')
    axes[0].set_title(f"Accuracies for CNN model on Catan tile set")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Plot train and validation losses
    axes[1].plot(hist.history["loss"], label="Train loss", linestyle='solid', color='b')
    axes[1].plot(hist.history["val_loss"], label="Validation loss", linestyle='dashed', color='c')
    axes[1].set_title(f"Losses for model on Catan tile set")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def to_tensor(tensor_str):
    # Convert string to a list using ast.literal_eval
    tensor_list = ast.literal_eval(tensor_str)
    return torch.tensor(tensor_list)

def model_eval(model, ds_test):

    ds_test = ds_test.batch(BATCH_SIZE)

    loss, acc = model.evaluate(ds_test, verbose=2)
    print(f'Model loss on the test dataset: {loss}')
    print(f'Model accuracy on the test set: {acc}')

def model_predict(model, sample, label_encoder, img_size):
    # Read the image
    img = Image.open(sample).convert('RGB')
    img_np = ToTensor()(img)
    img_np = torch.nn.functional.interpolate(img_np.unsqueeze(0), size=img_size[:2])
    img_np = img_np.permute(0, 2, 3, 1)
    img_np = img_np.numpy()

    pred = model.predict(img_np)
    print(pred)

    pred_label = label_encoder.inverse_transform([np.argmax(pred)])

    print(f'Image at path: {sample} is a {pred_label} tile')

def model_training(model, train_set, valid_set, epochs):
    # Stop when the loss does not improve significantly over 3 epochs
    callback = callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=3)

    # Keras expects a batched dataset
    batched_train = train_set.batch(BATCH_SIZE)
    batched_valid = valid_set.batch(BATCH_SIZE)

    # Fit the model to the training data
    hist = model.fit(x=batched_train, validation_data=batched_valid,  epochs=epochs, callbacks=[callback])

    # Plot training loss and accuracy
    plot_hist(hist)

    return model

def data_augmentation():
    # Data augmentation component
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.2),
    ])

def build_dataset(dataset_path, valid_split, test_split):

    print(f'Compiling dataset at path: {dataset_path}')

    with open(dataset_path, 'rb') as f:
        (X, y) = pickle.load(f)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    valid_size = int(valid_split * len(X))
    test_size = int(test_split * len(X))
    train_size = len(X) - valid_size - test_size

    final_indices = [indices[:train_size], indices[train_size:train_size + valid_size], indices[train_size + valid_size:]]

    X_train = tf.gather(X, final_indices[0])
    y_train = tf.gather(y, final_indices[0])

    X_valid = tf.gather(X, final_indices[1])
    y_valid = tf.gather(y, final_indices[1])

    X_test = tf.gather(X, final_indices[2])
    y_test = tf.gather(y, final_indices[2])

    return [(X_train, y_train), (X_valid, y_valid), (X_test, y_test)]


def build_cnn(input_shape):
    # Augment the training data
    augmentation = data_augmentation()

    # Define CNN model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        augmentation,
        layers.Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=input_shape),
        layers.MaxPool2D(pool_size=(1,1), padding='valid'),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    # Define optimizer
    optimizer = keras.optimizers.Adam(learning_rate=5e-5, use_ema=True)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)

    return model

if __name__ == "__main__":

    # Fix error where multiple libiomp5md.dll files are present
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    ##### PARAMETER DEFINITION #####

    #Downsample all images for faster training
    IMG_SIZE = (243, 256, 3)
    digit_size = (100, 100, 3)
    BATCH_SIZE = 32
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 100 # the maximum number of epochs used to train the model
    validation_split = 0.2
    test_split = 0.1
    path_to_predict = '../data/sample/test1.png'
    model_save_path = '../board_piece_classification/model/tile_detector_hexagons2.keras'
    dataset_path = '../data/full/compiled_dataset/synthetic_dataset_hexagons.pkl'
    label_encoder_path = '../data/full/compiled_dataset/label_encoder/label_encoder_hexagons.pkl'

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
    model = build_cnn(IMG_SIZE)

    # Then train it on the input data
    # This will also plot the train and validation accuracies
    model = model_training(model, train_set, validation_set, epochs)

    model.save(model_save_path)
    print(f'Trained model saved at: {model_save_path}')

    ##### TESTING THE MODEL #####

    # Test the model on the test set
    model_eval(model, test_set)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    # Predict a single sample that is different from the test set
    model_predict(model, path_to_predict, label_encoder, IMG_SIZE[:2])