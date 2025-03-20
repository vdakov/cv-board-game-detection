from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


def to_tf_dataset(ds_dict, output_path):
    input = [x.numpy() for x in ds_dict['img_tensor']]
    labels = ds_dict['img_label']

    # Convert labels (string categories) to integer labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Convert to TensorFlow tensors
    X_tensorflow = tf.convert_to_tensor(input, dtype=tf.float32)
    y_tensorflow = tf.convert_to_tensor(y_encoded, dtype=tf.int32)

    with open(f'{output_path}/labeled_synthetic_samples.pkl', 'wb') as f:
        pickle.dump((X_tensorflow.numpy(), y_tensorflow.numpy(), label_encoder), f)


def get_labeled_hexagons(tile_img_folder, mask_folder, ref_img_folder, output_path, resize_shape):

    transform = ToTensor()

    # Initialize the mask (used for labeling)
    mask = Image.open(f'{mask_folder}/hexagon_mask.jpg')
    mask_np = transform(mask)

    image_classes = ['clay', 'default', 'sheep', 'stone', 'wheat', 'wood']
    # Dictionary that stores the reference images
    ref_dict = {}
    # Dictionary that stores for each label all images fitting this label
    final_dict = {
        'img_path':[],
        'img_tensor':[],
        'img_label':[]
    }

    # Get reference images
    for cls in image_classes:
        img = Image.open(f'{ref_img_folder}/{cls}.png')
        img_np = transform(img)

        # interpolate to ensure same size
        resized_mask = interpolate(mask_np.unsqueeze(0), resize_shape).squeeze(0)
        resized_img = interpolate(img_np.unsqueeze(0), resize_shape).squeeze(0)

        masked_ref = resized_img * resized_mask

        ref_dict[cls] = masked_ref

    # Label images
    for i in range(19):
        img_path = f'{tile_img_folder}/hexagon_{i}.png'

        img = Image.open(img_path)
        img_np = transform(img)

        # interpolate to ensure same size
        resized_mask = interpolate(mask_np.unsqueeze(0), resize_shape).squeeze(0)
        resized_img = interpolate(img_np.unsqueeze(0), resize_shape).squeeze(0)

        masked_img = resized_img * resized_mask

        mse = np.inf
        final_class = ''
        for cls in image_classes:
            masked_ref = ref_dict[cls]

            current_mse = torch.mean((masked_img - masked_ref) ** 2)

            if current_mse < mse:
                mse = current_mse
                final_class = cls

        final_dict['img_path'].append(img_path)
        final_dict['img_tensor'].append(resized_img.permute(1, 2, 0))
        final_dict['img_label'].append(final_class)

    df = pd.DataFrame.from_dict(final_dict)
    df.to_csv(f'{output_path}/labeled_synthetic_samples.csv', index=False)

    return final_dict


if __name__ == "__main__":

    tile_img_folder = '../data/sample/mined_synthetic_tiles_sample'
    mask_folder = '../data/tile_masks'
    ref_img_folder = '../data/sample/synthetic_reference_tiles'
    output_path = '../data/sample'
    resize_shape = (135, 121)

    # Get masked hexagons
    final_dict = get_labeled_hexagons(tile_img_folder, mask_folder, ref_img_folder, output_path, resize_shape)
    to_tf_dataset(final_dict, output_path, 3)