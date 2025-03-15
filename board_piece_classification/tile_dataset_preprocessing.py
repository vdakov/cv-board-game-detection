from PIL import Image
import numpy as np

def get_tile_images(tile_img_folder, mask_folder, is_hexagon=True):

    # Initialize the mask
    if is_hexagon:
        mask = Image.open(f'{mask_folder}/hexagon_mask.jpg')
    else:
        mask = Image.open(f'{mask_folder}/circle_mask.jpg')
    mask_np = np.array(mask)

    for i in range(19):
        img = Image.open(f'{tile_img_folder}/hexagon_{i}.png')
        img_np = np.array(img)

        # Apply the mask
        masked_img = np.multiply(img_np, mask_np)


if __name__ == "__main__":

    tile_img_folder = 'data/sample/mined_synthetic_tiles_sample'

    # Get masked hexagons
    hexagons = get_tile_images(tile_img_folder, is_hexagon=True)

    # Get masked numbers
