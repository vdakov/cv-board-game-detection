import keras
from keras import layers
import matplotlib.pyplot as plt

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

def model_eval(model, ds_test):
    loss, acc = model.evaluate(ds_test, verbose=2)
    print(f'Model loss on the test dataset: {loss}')
    print(f'Model accuracy on the test set: {acc}')

def model_training(model, ds_train, epochs, ds_valid):
    hist = model.fit(ds_train, epochs=epochs, validation_data=ds_valid)
    plot_hist(hist)
    return model

def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
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
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return built_model

if __name__ == "__main__":

    ##### PARAMETER DEFINITION #####
    # EfficientNetB0 expects images of shape 224 x 224
    IMG_SIZE = 224
    BATCH_SIZE = 64
    NUM_CLASSES = 6  # there are six tile types in Catan
    epochs = 50 # the number of epochs used to train the model

    ##### DATASET PRE-PROCESSING #####
    ds_train = None
    ds_valid = None
    ds_test = None

    ##### TRAINING THE MODEL #####

    # First build the model
    model = build_model()

    # Then train it on our input data
    # This will also plot the train and validation accuracies
    model = model_training(model, ds_train, epochs, ds_valid)

    # Test the model on the test set
    model_eval(model, ds_test)