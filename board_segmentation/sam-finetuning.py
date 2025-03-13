import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import matplotlib.pyplot as plt

from mask_extraction import get_args
import cv2
from torch.nn.functional import threshold, normalize
import numpy as np
import pickle

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

def plot_masks(gt_masks, obtained_masks):
    _, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].imshow(gt_masks)
    ax[1].imshow(obtained_masks)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = get_args()
    im_folder = args.board_directory

    sam_model = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint_path)

    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters())
    loss_fn = torch.nn.MSELoss()

    num_epochs = 50

    PATH = f'sam_checkpoint/finetuned_sam.pth'

    args = get_args()
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    image_emb_tensors = []
    bounding_boxes_tensors = []
    input_img_shapes = []
    ground_truth_mask_list = []

    # Calculate and store all image embeddings to speed up compute time
    for i in range(11):
        image_path = f"{im_folder}/canvas_image_{i}.png"
        input_image = cv2.imread(image_path, flags=cv2.COLOR_BGR2RGB)
        img_size = input_image.shape[:2][::-1]  # get the original image height and width as (H, W)
        input_img_shapes.append(img_size)

        # get the saved ground truth masks
        ground_truth_masks = get_object_from_file(
            f'{args.sam_finetune_save_dir}/image_{i}_outputs/ground_truth_masks.pkl')
        ground_truth_mask_list.append(ground_truth_masks)

        # further preprocessing of the input image
        input_image = cv2_to_tensor(input_image)

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)  # input image
            image_emb_tensors.append(image_embedding)


        bounding_boxes = get_object_from_file(f'{args.sam_finetune_save_dir}/image_{i}_outputs/bounding_boxes.pkl')
        bounding_boxes = get_bounding_boxes(bounding_boxes)
        bounding_boxes_tensors.append(bounding_boxes)


    for epoch in range(num_epochs):
        print(f"Reached epoch {epoch}/{num_epochs - 1}")

        # Batched optimization for 11 images
        for i in range(11):
            print(f"Reached image {i}/10")
            # optimize for a given number of epochs

            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=bounding_boxes_tensors[i], # bounding box coord converted to pytorch tensors
                    masks=None,
                )

            low_res_masks, iou_predictions = sam_model.mask_decoder(
              image_embeddings=image_emb_tensors[i],
              image_pe=sam_model.prompt_encoder.get_dense_pe(),
              sparse_prompt_embeddings=sparse_embeddings,
              dense_prompt_embeddings=dense_embeddings,
              multimask_output=False,
            )
            low_res_masks = low_res_masks.permute(1, 0, 2, 3)

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, image_emb_tensors[i].shape, input_img_shapes[i]).squeeze(0).to(device)

            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.as_tensor(ground_truth_mask_list[i] > 0, dtype=torch.float32).to(device)

            loss = loss_fn(binary_mask, gt_mask_resized)

            print(f"\tEpoch {epoch} image {i} loss {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save the finetuned model to a new path
    torch.save(sam_model.state_dict(), PATH)