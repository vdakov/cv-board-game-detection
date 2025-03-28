import tensorflow as tf
import keras
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
import pickle

def preprocess_image(img, zoom, final_size, threshold):
    w, h = img.size

    # First zoom to the center of the image
    x = w / 2
    y = h / 2

    zoom2 = zoom * 2

    img = img.crop((x - w / zoom2, y - h / zoom2,
                    x + w / zoom2, y + h / zoom2))

    img = img.resize(final_size)

    # # Then keep the number only (so anything that is either black or bright red)
    # # and convert the rest to white
    # img = img.convert('L')
    # img = img.point(lambda p: 255 if p > threshold else 0)
    # img = img.convert('1')

    img = img.convert('RGB')

    img.show()

    return img

def model_predict(model, sample, label_encoder, img_size, is_number):
    # Read the image
    img = Image.open(sample)
    if is_number:
        img = preprocess_image(img, 4, img_size, 85)
    img_np = ToTensor()(img)
    img_np = torch.nn.functional.interpolate(img_np.unsqueeze(0), size=img_size[:2])
    img_np = img_np.permute(0, 2, 3, 1)
    img_np = img_np.numpy()

    pred = model.predict(img_np)
    print(pred)

    pred_label = label_encoder.inverse_transform([np.argmax(pred)])

    print(f'Image at path: {sample} is a {pred_label} tile')

if __name__ == '__main__':

    model_path = 'model/tile_detector_numbers.keras'
    img_path = '../data/sample/test.jpg'
    label_encoder_path = '../data/full/compiled_dataset/label_encoder/label_encoder_numbers.pkl'
    IMG_SIZE = (100, 100, 3)

    model = keras.models.load_model(model_path)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

    model_predict(model, img_path, label_encoder, IMG_SIZE[:2], True)

