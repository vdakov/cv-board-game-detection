from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def cluster_masks(anns, min_samples=3, eps=0.5):
    features = np.array([ann['area'] for ann in anns]).reshape(-1, 1)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features_scaled)

    for i, ann in enumerate(anns):
        ann['cluster'] = clustering.labels_[i]

    return anns


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint_path", help="Echo the path to the SAM checkpoint here.", type=str, default="sam_checkpoint/sam_vit_h_4b8939.pth")
    parser.add_argument("--model_name", help="Echo the name of the model you want to use", type=str, default="vit_h")
    parser.add_argument("--board_directory", help="Echo the path to the directory with the board images", type=str, default="../catan_data/mined_synthetic_boards_sample/")
    parser.add_argument("--save_dir", help="Echo the path to the directory where you want your hexagons saved", type=str, default="../catan_data/mined_synthetic_tiles_sample/")
    args = parser.parse_args()
    return args

def show_anns(original, anns, cluster=False):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    _, ax = plt.subplots(1, 3, figsize=(15,5))
    if cluster:
        sorted_anns = cluster_masks(sorted_anns)


    H, W = sorted_anns[0]['segmentation'].shape
    img = np.zeros((H, W, 4))  # Start with black image (RGBA)
    
    for ann in sorted_anns:
        m = ann['segmentation']  
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m, :] = color_mask  #

    cluster_img = np.zeros((H, W, 4))
    if cluster:
        unique_clusters = set(ann['cluster'] for ann in sorted_anns if ann['cluster'] != -1)
        print("Estimated number of clusters: %d" % len(unique_clusters))
        print("Estimated number of noise points: %d" % len(set(ann['cluster'] for ann in sorted_anns if ann['cluster'] == -1)))
        cluster_colors = {c: np.concatenate([np.random.random(3), [0.7]]) for c in unique_clusters}  

        for ann in sorted_anns:
            cluster_label = ann['cluster']
            if cluster_label != -1:  # Ignore outliers
                m = ann['segmentation']
                cluster_img[m, :] = cluster_colors[cluster_label]


    ax[0].imshow(original)
    ax[1].imshow(img)
    ax[2].imshow(cluster_img)
    plt.axis('off')
    plt.show()

    return cluster_img

if __name__ == "__main__":
    args = get_args()
    im_folder = args.board_directory
    sam_checkpoint = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint_path)
    mask_generator = SamAutomaticMaskGenerator(sam_checkpoint)
    for i in range(1):
        image_path = f"{im_folder}/canvas_image_{i}.png"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        img = cv2.GaussianBlur(img, (5, 5), 0)
        

        masks = mask_generator.generate(img)
        
        cluster_img = show_anns(img, masks, cluster=True)
        cluster_img = (cluster_img * 255).astype(np.uint8)
        cluster_img = cv2.cvtColor(cluster_img, cv2.COLOR_RGB2GRAY)

        contours, _ = cv2.findContours(cluster_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hexagons = []
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

            if len(approx) == 6:  # Hexagon has six sides
                hexagons.append(approx)

        # Draw detected hexagons
        result = img.copy()
        cv2.drawContours(result, hexagons, -1, (0, 255, 0), 3)

        # Plot the result
        plt.figure(figsize=(10, 5))
        plt.imshow(result, cmap='gray')
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("Detected Hexagons")
        plt.axis("off")
        plt.show()

        save_dir = "hexagon_outputs"  # Change this to your desired directory
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

        for idx, hexagon in enumerate(hexagons):
            x, y, w, h = cv2.boundingRect(hexagon)  # Get bounding box
            hex_crop = img[y:y+h, x:x+w]  # Crop the region

            save_path = os.path.join(args.save_dir, f"hexagon_{idx}.png")
            cv2.imwrite(save_path, cv2.cvtColor(hex_crop, cv2.COLOR_RGB2BGR))  # Save the hexagon
            



   
