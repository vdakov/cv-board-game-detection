import keras
from PIL import Image
from PIL.Image import Resampling
from torchvision.transforms import ToTensor
import torch
import numpy as np
import pickle
import pytesseract
import re


def predict_hexagons(model, img, label_encoder, img_size):
    # Read the image
    img_np = ToTensor()(img)
    img_np = torch.nn.functional.interpolate(img_np.unsqueeze(0), size=img_size[:2])
    img_np = img_np.permute(0, 2, 3, 1)
    img_np = img_np.numpy()

    pred = model.predict(img_np)

    pred_label = label_encoder.inverse_transform([np.argmax(pred)])

    return pred_label


def preprocess_image(img, zoom, threshold):
    w, h = img.size

    # First zoom to the center of the image
    x = w / 2
    y = h / 2

    zoom2 = zoom * 2

    img = img.crop((x - w / zoom2, y - h / zoom2, x + w / zoom2, y + h / zoom2))

    img = img.resize((200, 200), resample=Resampling.LANCZOS)

    # Then keep the number only (so anything that is either black or bright red)
    # and convert the rest to white
    img = img.convert("L")
    img = img.point(lambda p: 255 if p > threshold else 0)
    img = img.convert("1")

    return img


def predict_number(img):
    img1 = np.array(preprocess_image(img, 3.5, 95))
    text = pytesseract.image_to_string(img1, config="--psm 13")

    number = [int(s) for s in re.findall(r"\b\d+\b", text)]

    return str(number[0]) if len(number) > 0 else '0'


def predict_image(img, models, label_encoder, IMG_SIZE):
    hex_label = predict_hexagons(models, img, label_encoder, IMG_SIZE[:2])

    number_label = predict_number(img)

    return hex_label, number_label


if __name__ == "__main__":

    # Attempt to predict the tile and number of a few sample tiles generated synthetically
    model_path = "data/models/tile_detector_hexagons.keras"
    img_folder_path = "../data/sample/mined_synthetic_tiles_sample"
    label_encoder_path = "data/output/compiled_dataset/label_encoder/label_encoder_hexagons.pkl"
    IMG_SIZE = (100, 100, 3)

    model = keras.models.load_model(model_path)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    for i in range(0, 19):
        img_path = f"{img_folder_path}/hexagon_{i}.png"

        img = Image.open(img_path).convert('RGB')

        hex_label, number_label = predict_image(
            img, model, label_encoder, IMG_SIZE
        )

        print(f"Hex at path {img_path} is of type {hex_label} - {number_label}")
