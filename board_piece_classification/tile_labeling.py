from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from torch.nn.functional import interpolate
import pandas as pd

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
        final_dict['img_tensor'].append(img_np)
        final_dict['img_label'].append(final_class)

    df = pd.DataFrame.from_dict(final_dict)
    df.to_csv(f'{output_path}/labeled_synthetic_samples.csv')


if __name__ == "__main__":

    tile_img_folder = '../data/sample/mined_synthetic_tiles_sample'
    mask_folder = '../data/tile_masks'
    ref_img_folder = '../data/sample/synthetic_reference_tiles'
    output_path = '../data/sample'
    resize_shape = (135, 121)

    # Get masked hexagons
    get_labeled_hexagons(tile_img_folder, mask_folder, ref_img_folder, output_path, resize_shape)