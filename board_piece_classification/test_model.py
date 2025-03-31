import tensorflow as tf
import keras
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
import pickle

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

if __name__ == '__main__':

    model_path = 'model/tile_detector_hexagons2.keras'
    img_folder_path = '../data/sample'
    label_encoder_path = '../data/full/compiled_dataset/label_encoder/label_encoder_hexagons.pkl'
    IMG_SIZE = (243, 256, 3)

    model = keras.models.load_model(model_path)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    for i in range(1, 5):
        img_path = f'{img_folder_path}/test{i}.png'
        model_predict(model, img_path, label_encoder, IMG_SIZE[:2])

