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

    IMG_SIZE = (135, 121, 3)
    BATCH_SIZE = 64
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 100 # the maximum number of epochs used to train the model
    validation_split = 0.2
    test_split = 0.1
    train_set_path = '../data/full/compiled_dataset/mined_synthetic_samples.pkl'
    path_to_predict = '../data/sample/test.jpg'
    model_save_path = '../board_piece_classification/model/tile_detector.keras'

    ##### DATASET PRE-PROCESSING #####

    with open(train_set_path, 'rb') as f:
        X, y, label_encoder = pickle.load(f)

    # Convert to TensorFlow tensors
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X))

    test_size = int(test_split * len(X))
    validation_size = int(validation_split * len(X))
    train_size = len(X) - test_size - validation_size

    train_set = dataset.take(train_size)
    validation_set = dataset.skip(validation_size)
    test_set = dataset.skip(test_size)

    ##### TRAINING THE MODEL #####

    # First build the model
    model = build_cnn()

    # Then train it on the input data
    # This will also plot the train and validation accuracies
    model = model_training(model, train_set, validation_set, epochs)

    model.save(model_save_path)
    print(f'Trained model saved at: {model_save_path}')

    ##### TESTING THE MODEL #####

    # Test the model on the test set
    model_eval(model, test_set)

    # Predict a single sample that is different from the test set
    model_predict(model, path_to_predict, label_encoder, IMG_SIZE)