import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_object_from_file(filepath):
    with open(filepath, 'rb') as f:
        loaded_masks = pickle.load(f)
    return loaded_masks

sam_finetune_save_dir = '../catan_data/sam_finetuning_input_data/'

ground_truth_masks = get_object_from_file(
            f'{sam_finetune_save_dir}/image_{0}_outputs/ground_truth_masks.pkl')

print(ground_truth_masks[3].shape)

gt_stack = np.logical_or.reduce(ground_truth_masks, axis=0).astype(np.uint8)

plt.figure(figsize=(6, 6))
plt.imshow(gt_stack, cmap='binary')
plt.colorbar(label='Mask Value')
plt.title('Binary Mask Visualization')
plt.show()