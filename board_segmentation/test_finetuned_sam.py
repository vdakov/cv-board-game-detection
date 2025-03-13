from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_masks(masks, image):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    print('Plotted image')

    H, W = masks[0]['segmentation'].shape
    img = np.zeros((H, W, 4))  # Start with black image (RGBA)

    for mask in masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m, :] = color_mask

    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    path_to_sam = 'sam_checkpoint/finetuned_sam.pth'

    sam_checkpoint = sam_model_registry["vit_h"](checkpoint=path_to_sam)
    mask_generator = SamAutomaticMaskGenerator(sam_checkpoint)

    print('Loaded new SAM model')

    image_path = f"../catan_data/sam_finetuning_test_data/catan_board1.jpg"
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    print('Loaded image to get the masks for')

    masks = mask_generator.generate(img)

    print(f'Generated masks for image at path {image_path}')

    plot_masks(masks, img)