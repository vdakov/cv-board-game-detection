import keras
from keras import layers, models, callbacks
import matplotlib.pyplot as plt
import torch
import ast
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.python.keras.layers import MaxPool2D
from torchvision.transforms import ToTensor


def plot_hist(hist):
    fig, axes = plt.subplots(1, 2)

    axes[0].plot(hist.history["accuracy"])
    axes[0].set_title(f"Training accuracy for model: ")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Epoch")

    axes[1].plot(hist.history["loss"])
    axes[1].set_title(f"Training loss for model: ")
    axes[1].set_ylabel("Loss")
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.show()

def to_tensor(tensor_str):
    # Convert string to a list using ast.literal_eval
    tensor_list = ast.literal_eval(tensor_str)
    return torch.tensor(tensor_list)

def model_eval(model, ds_test):
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

def model_training(model, X, y, epochs):
    # Stop when the loss does not improve significantly over 3 epochs
    callback = callbacks.EarlyStopping(monitor='loss', min_delta=0.01, patience=3)

    # Fit the model to the training data
    hist = model.fit(x=X, y=y, epochs=epochs, callbacks=[callback])

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

def build_cnn():
    # Augment the training data
    augmentation = data_augmentation()

    # Define CNN model
    model = models.Sequential([
        layers.Input(shape=IMG_SIZE),
        augmentation,
        layers.Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=IMG_SIZE),
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

    ##### PARAMETER DEFINITION #####

    # EfficientNetB0 expects images of shape 224 x 224
    IMG_SIZE = (135, 121, 3)
    BATCH_SIZE = 64
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 100 # the number of epochs used to train the model
    dataset_path = '../data/sample/labeled_synthetic_samples.pkl'
    path_to_predict = '../data/sample/test2.png'
    model_save_path = '../data/sample/tile_detector.keras'

    ##### DATASET PRE-PROCESSING #####

    with open(dataset_path, 'rb') as f:
        X_data, y_labels, label_encoder = pickle.load(f)

    # Convert to TensorFlow tensors if needed
    X_tensorflow = tf.convert_to_tensor(X_data, dtype=tf.float32)
    y_tensorflow = tf.convert_to_tensor(y_labels, dtype=tf.int32)

    ##### TRAINING THE MODEL #####

    # First build the model
    model = build_cnn()

    # Then train it on our input data
    # This will also plot the train and validation accuracies
    model = model_training(model, X_tensorflow, y_tensorflow, epochs)

    model.save(model_save_path)
    print(f'Trained model saved at: {model_save_path}')

    ##### TESTING THE MODEL #####

    # Test the model on the test set
    model_predict(model, path_to_predict, label_encoder, IMG_SIZE)