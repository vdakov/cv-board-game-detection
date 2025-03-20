import keras
from keras import layers
import matplotlib.pyplot as plt
import torch
import ast
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    # plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
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
    model.fit(x=X, y=y, epochs=epochs)
    # plot_hist(hist)
    return model

def build_model():
    inputs = layers.Input(shape=IMG_SIZE)
    built_model = keras.applications.EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    built_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(built_model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

    # Compile
    built_model = keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = keras.optimizers.Adam(learning_rate=1e-2)
    built_model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return built_model

if __name__ == "__main__":

    ##### PARAMETER DEFINITION #####

    # EfficientNetB0 expects images of shape 224 x 224
    IMG_SIZE = (135, 121, 3)
    BATCH_SIZE = 64
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 50 # the number of epochs used to train the model
    dataset_path = '../data/sample/labeled_synthetic_samples.pkl'
    path_to_predict = '../data/sample/test.jpg'

    ##### DATASET PRE-PROCESSING #####

    with open(dataset_path, 'rb') as f:
        X_data, y_labels, label_encoder = pickle.load(f)

    # Convert to TensorFlow tensors if needed
    X_tensorflow = tf.convert_to_tensor(X_data, dtype=tf.float32)
    y_tensorflow = tf.convert_to_tensor(y_labels, dtype=tf.int32)

    ##### TRAINING THE MODEL #####

    # First build the model
    model = build_model()

    # Then train it on our input data
    # This will also plot the train and validation accuracies
    model = model_training(model, X_tensorflow, y_tensorflow, epochs)

    ##### TESTING THE MODEL #####

    # Test the model on the test set
    model_predict(model, path_to_predict, label_encoder, IMG_SIZE)