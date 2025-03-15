import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt

from mask_extraction import get_args
import cv2
from torch.nn.functional import threshold, normalize
import numpy as np
import pickle
from PIL import Image

def get_bounding_boxes(bounding_box_dict):
    """"
    For a given image of a Catan board, gets the bounding boxes of all of the identified hexagons.
    @:param bounding_boxes_filepath: filepath of the bounding boxes for one board image.
    @:return a dict where the keys are the index of the hexagon and the values are Torch tensors containing the
    bounding box coordinates
    """

    bounding_boxes = []

    for k in bounding_box_dict.keys():
        row = bounding_box_dict[k]

        box_tensor = torch.tensor(row).unsqueeze(0)

        bounding_boxes.append(box_tensor)

    return torch.stack(bounding_boxes, dim=0)

def cv2_to_tensor(image):
    """"
    Transform an image imported with opencv to a Tensor.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_img = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_img, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    input_img = sam_model.preprocess(transformed_image)
    return input_img

def get_object_from_file(filepath):
    with open(filepath, 'rb') as f:
        loaded_masks = pickle.load(f)
    return loaded_masks

def load_everything_from_files():

    bounding_boxes_tensors = []
    ground_truth_mask_list = []

    # Calculate and store all image embeddings to speed up compute time
    for i in range(11):

        img_data_folder = f'{args.sam_finetune_save_dir}/image_{i}_outputs'

        # get the saved ground truth masks
        ground_truth_masks = get_object_from_file(f'{img_data_folder}/ground_truth_masks.pkl')
        ground_truth_mask_list.append(ground_truth_masks)

        bounding_boxes = get_object_from_file(f'{img_data_folder}/bounding_boxes.pkl')
        bounding_boxes = get_bounding_boxes(bounding_boxes)
        bounding_boxes_tensors.append(bounding_boxes)

    return bounding_boxes_tensors, ground_truth_mask_list

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

    ground_truth_masks = torch.tensor(np.array([circle_mask, hexagon_mask]))

    bounding_boxes = {
        '0': [37, 32, 97, 89], # circle bounding box
        '1': [0, 0, 136, 120]} # hexagon bounding box

    return ground_truth_masks, bounding_boxes

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    im_folder = '../catan_data/sam_finetuning_input_data/image_0_outputs'

    sam_model = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint_path)

    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
    loss_fn = torch.nn.MSELoss()

    num_epochs = 10

    PATH = f'sam_checkpoint/finetuned_sam.pth'

    args = get_args()

    ground_truth_mask_list, bounding_boxes = (
        get_masks_bounding_boxes('../catan_data/sam_finetuning_input_data/image_0_outputs'))

    bounding_boxes_tensors = get_bounding_boxes(bounding_boxes)

    image_path = f"{im_folder}/img_0_hexagon_{0}.png"
    input_image = cv2.imread(image_path, flags=cv2.COLOR_BGR2RGB)
    img_size = input_image.shape[:2][::-1]  # get the original image height and width as (H, W)

    # optimize for a given number of epochs
    for epoch in range(num_epochs):
        print(f"Reached epoch {epoch}/{num_epochs - 1}")

        # Batched optimization for 11 images
        for i in range(19):
            print(f"Reached image {i}/10")

            image_path = f"{im_folder}/img_0_hexagon_{i}.png"
            input_image = cv2.imread(image_path, flags=cv2.COLOR_BGR2RGB)

            # further preprocessing and embedding of the input image
            input_image = cv2_to_tensor(input_image)

            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)  # input image

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bounding_boxes_tensors, # bounding box coord converted to pytorch tensors
                    masks=None,
                )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
              image_embeddings=image_embedding,
              image_pe=sam_model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=False,
            )

            low_res_masks = low_res_masks.permute(1, 0, 2, 3)

            upscaled_masks = (sam_model.postprocess_masks(low_res_masks, image_embedding.shape, img_size).squeeze(0)
                              .to(device))

            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = (torch.as_tensor(ground_truth_mask_list > 0, dtype=torch.float32).permute(0, 2, 1)
                               .to(device))

            loss = loss_fn(binary_mask, gt_mask_resized)

            print(f"\tEpoch {epoch} image {i} loss {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save the finetuned model to a new path
    torch.save(sam_model.state_dict(), PATH)