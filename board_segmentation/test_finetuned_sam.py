import torch
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt
import cv2
from torch.nn.functional import threshold, normalize
import numpy as np
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F


def plot_multiple_masks(image, masks, colors=None, alpha=0.5):
    """
    Plot multiple binary masks with different colors on the original image.

    Parameters:
    - image: Original image
    - masks: List of binary masks
    - colors: List of RGB colors for each mask, default is randomly generated
    - alpha: Transparency of the overlays
    """
    if colors is None:
        colors = [np.random.rand(3) for _ in range(len(masks))]

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print(masks.shape)

    plt.figure(figsize=(5, 5))

    for mask, color in zip(masks, colors):
        mask_colors = np.array([[0, 0, 0, 0], list(color) + [alpha]])
        mask_cmap = LinearSegmentedColormap.from_list('custom_cmap', mask_colors, N=2)
        plt.imshow(mask, cmap=mask_cmap)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_bounding_boxes(bounding_box_dict):
    """
    For a given image of a Catan board, gets the bounding boxes of all of the identified hexagons.
    @:param bounding_box_dict: dictionary of the bounding boxes for one board image.
    @:return a tensor containing the bounding box coordinates
    """
    bounding_boxes = []
    for k in bounding_box_dict.keys():
        row = bounding_box_dict[k]
        box_tensor = torch.tensor(row).unsqueeze(0)
        bounding_boxes.append(box_tensor)
    return torch.stack(bounding_boxes, dim=0)

def cv2_to_tensor(image, device):
    """
    Transform an image imported with opencv to a Tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_img = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_img, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_img = sam_model.preprocess(transformed_image)
    return input_img

def jpg_to_binary_mask(mask_data_file):
    # Load the image
    mask_img = Image.open(mask_data_file).convert('L')  # Convert to grayscale
    # Convert to numpy array (0-255)
    mask_array = np.array(mask_img)
    mask = (mask_array > 1).astype(np.float32)  # 1 for white, 0 for black
    return torch.from_numpy(mask)

def get_masks_bounding_boxes(mask_folder):
    circle_mask = jpg_to_binary_mask(f'{mask_folder}/circle_mask.jpg')
    hexagon_mask = jpg_to_binary_mask(f'{mask_folder}/hexagon_mask.jpg')
    ground_truth_masks = torch.stack([circle_mask, hexagon_mask], dim=0)
    bounding_boxes = {
        '0': [37, 32, 97, 89], # circle bounding box
        '1': [0, 0, 136, 120]  # hexagon bounding box
    }
    return ground_truth_masks, bounding_boxes

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    path_to_sam = 'sam_checkpoint/sam_vit_h_4b8939.pth'

    sam_model = sam_model_registry["vit_h"](checkpoint=path_to_sam)

    print('Loaded SAM model')

    im_folder = '../catan_data/sam_finetuning_input_data/image_0_outputs'

    image_path = f"{im_folder}/img_0_hexagon_0.png"
    orig_image = cv2.imread(image_path, flags=cv2.COLOR_BGR2RGB)
    img_height, img_width = orig_image.shape[:2]

    ground_truth_mask_list, bounding_boxes = (
        get_masks_bounding_boxes('../catan_data/sam_finetuning_input_data/image_0_outputs'))

    bounding_boxes_tensors = get_bounding_boxes(bounding_boxes).squeeze(1)

    # further preprocessing and embedding of the input image
    input_image = cv2_to_tensor(orig_image, device)

    image_embedding = sam_model.image_encoder(input_image)  # input image

    print('Obtained image embedding')

    # Get SAM's expected input resolution
    input_size = sam_model.image_encoder.img_size  # Typically 1024

    # Resize your ground truth masks to match the input image dimensions
    resized_masks = F.interpolate(
        ground_truth_mask_list.unsqueeze(1).float(),  # Add channel dim
        size=(input_size, input_size),
        mode='bilinear',
        align_corners=False
    )
    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=None,
        boxes=None,
        masks=resized_masks
    )

    print('Obtained embeddings via prompt encoder')

    # Now these should be compatible for the mask decoder
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=sam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    print('Obtained masks via mask decoder')
    print(low_res_masks.shape)
    print(input_image.shape)

    low_res_masks = low_res_masks.permute(1, 0, 2, 3)

    upscaled_masks = F.interpolate(
        low_res_masks,
        size=(img_height, img_width),
        mode='bilinear',
        align_corners=False
    )

    # 8. Apply threshold to get binary masks
    binary_masks = (upscaled_masks > 0.0).float()

    print(f'Generated masks for image at path {image_path}')

    plot_multiple_masks(orig_image, binary_masks.detach().squeeze(0))