from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

def to_tf_dataset(ds_dict, output_path):
    input = [x.numpy() for x in ds_dict['img_tensor']]
    labels = ds_dict['img_label']

    # Convert labels (string categories) to integer labels
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit(labels)

    # Save the employed label encoder to file
    with open(f'{output_path}/label_encoder/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    y_encoded = label_encoder.transform(labels)

    # Convert to TensorFlow tensors
    X_tensorflow = tf.convert_to_tensor(input, dtype=tf.float32)
    y_tensorflow = tf.convert_to_tensor(y_encoded, dtype=tf.int32)

    with open(f'{output_path}/mined_synthetic_dataset.pkl', 'wb') as f:
        pickle.dump((X_tensorflow.numpy(), y_tensorflow.numpy()), f)


def get_labeled_hexagons(tile_img_folder, mask_folder, ref_img_folder, output_path, resize_shape):
    transform = ToTensor()

    # Initialize the mask (used for labeling)
    mask = Image.open(f'{mask_folder}/hexagon_mask.jpg').convert('RGB')
    mask_np = transform(mask)

    image_classes = ['brick', 'desert', 'sheep', 'ore', 'wheat', 'lumber']

    # Dictionary that stores the reference images
    ref_dict = {}
    # Dictionary that stores for each label all images fitting this label
    final_dict = {
        'img_path': [],
        'img_tensor': [],
        'img_label': [],
        'closest_ref': []
    }

    # Get reference images
    for cls in image_classes:
        ref_dict[cls] = []
        for i in range(1, 4):
            img = Image.open(f'{ref_img_folder}/{cls}_{i}.png').convert('RGB')
            img_np = transform(img)

            # interpolate to ensure same size
            resized_mask = interpolate(mask_np.unsqueeze(0), resize_shape).squeeze(0)
            resized_img = interpolate(img_np.unsqueeze(0), resize_shape).squeeze(0)

            masked_ref = resized_img * resized_mask
            ref_dict[cls].append(masked_ref)

    # Label images
    for img_path in os.listdir(tile_img_folder):

        img = Image.open(f'{tile_img_folder}/{img_path}').convert('RGB')
        img_np = transform(img)

        # interpolate to ensure same size
        resized_mask = interpolate(mask_np.unsqueeze(0), resize_shape).squeeze(0)
        resized_img = interpolate(img_np.unsqueeze(0), resize_shape).squeeze(0)

        masked_img = resized_img * resized_mask

        mse = np.inf
        final_class = ''
        closest_ref_index = -1

        for cls in image_classes:
            for j, masked_ref in enumerate(ref_dict[cls]):
                current_mse = torch.mean(torch.abs(masked_img - masked_ref))

                if current_mse < mse:
                    mse = current_mse
                    final_class = cls
                    closest_ref_index = j + 1

        final_dict['img_path'].append(img_path)
        final_dict['img_tensor'].append(resized_img.permute(1, 2, 0))
        final_dict['img_label'].append(final_class)
        final_dict['closest_ref'].append(f'{final_class}_{closest_ref_index}')

        if i % 50 == 0:
            print(f'Reached image at path: {img_path} with label: {final_class} and closest index: {closest_ref_index}')

    df = pd.DataFrame.from_dict(final_dict)
    df.to_csv(f'{output_path}/labeled_synthetic_samples.csv', index=False)

    return final_dict


if __name__ == "__main__":

    tile_img_folder = '../data/full/mined_synthetic_tiles_blended'
    mask_folder = '../data/tile_masks'
    ref_img_folder = '../data/tile_datasets/hexagons'
    output_path = '../data/full/compiled_dataset'
    resize_shape = (100, 100)

    # Get masked hexagons
    final_dict = get_labeled_hexagons(tile_img_folder, mask_folder, ref_img_folder, output_path, resize_shape)
    to_tf_dataset(final_dict, output_path)