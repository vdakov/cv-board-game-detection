# cv-board-game-detection

A computer vision system that detects, segments, and classifies Catan game boards from image. The system identifies the board's position, correct for perspective distortion, segments individual hexagon tiles, and classifies them by resource type and number label. A visualizer and synthetic data miner are also implemented.

## Pipeline steps

1. **Homography Correction:** In a picture of a Catan Board is looked at from a certain perspective and correspondingly the board is distorted via some way. This distortion can be looked at as a combination of shears, translations and rotations. In other words, it can be expressed as a matrix $\mathbf H$, called a **homography matrix.** Correspondingly, this transformation can be inverted and a board can be looked at from some “default” perspective. In our case, this is top-down. A neural network predicts $\mathbf H$ and then $\mathbf H^{-1}$ is computed and applied to the image.

2. **Board Detection and Cropping:** To best-recognize the board, after correcting the perspective, we need to focus on it. We use YOLO object detection to find a bounding box over our board.

3. **Board/Hexagon Segmentation:** We then segment the board into separate tiles using a SAM model. We combine neighboring segments using DBScan, find all resulting contours, and filter them based on proportion-based heuristics of an expected proper Catan board (forms hexagons, proper size in relation to the imag, etc).

4. **Tile Classification:** We classify each hex tile separately, as a result of cropping the input image based on the found hexagonally-shaped contours. We use a CNN to recognize the resource type of the tile, and an OCR for the number labels. The two are separated using masks.
    
5. **Board Assembly:** Using the classification data for each tile, and the segmented hexes for their relative positions, we reconstruct the board into a game state, of a format suitable for both JSON and visualization.

## Quick structure overview

We built a modular pipeline for each step of recognition. For each component, we have wrapped all relevant functions to be used for end-to-end processing (`pipeline.py`). The `main` function of each component runs the training and/or evaluation for said component individually.  

For each component, we have a `data` folder that contains `input`/`output` folders used for training/evaluation, as well as a `models` folder for any weights/checkpoints/encodings.

## How to run

As a first step, set up a virtual environment with all dependencies found on `requirements.txt`. **Make sure your Python is old enough** (e.g. Python 3.9). 

Option 1 - Conda:

```
<!-- Make sure you are in the main directory. -->
conda create -n custom-env python=3.9
conda activate custom-env 
conda install pip 
pip install -r requirements.txt 
```

Option 2 - venv:

```bash
python -m venv venv
```
Activate the environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

You also need to install Tesseract OCR:
- Windows: Download and install from [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- Mac: `brew install tesseract`

Make sure the Tesseract OCR is in PATH for the scripts to access. Afterwards, you may rerun `pip install pytesseract` to ensure proper integration.

Then, we need to load all models used by the pipeline. We have already provided our best trained models for each task as part of the repo using Git LFS. Only exception is the SAM model, which will be downloaded automatically when running the scripts initially. Below we describe how one can train their own versions of the models.

### Perspective Correction Model

1. Prepare training data:
   - Place your training images in `pre_processing/data/output/perspective_distorted_boards`. Alternatively, you can synthetically generate data by running the `board_mining.py` script, followed by the `perspective_warping.py` script for preprocessing the samples.
   - Ensure you have corresponding bounding box coordinates in `bbox_coordinates.json`
2. Train the model using `homomography_training.py`. The model will be saved in `runs/models/homomography_hybrid_128_model.pth` which can be moved to its `models` folder.

### YOLO Board Detection Model
1. Convert training data to YOLO format using `to_yolo_format.py`.
2. Train the model using `yolo_training.py`. The model will be saved in `board_detection/data/output/train/weights/best.pt`, which can then be moved to its `models` folder.

### Tile Classification Model
1. Place your training images (cropped hex tiles) in `board_piece_classification/data/input`. Ensure images are organized by class in subdirectories.
2. Train the classification model using `train_classifier.py`. The model will be saved in `board_piece_classification/data/models/tile_detector_hexagons.keras`, along with its encoders.

## Running the Pipeline

After setting up all models, you can run the full pipeline on an image using `python pipeline.py --img_path path/to/your/image.jpg`.

The output will be saved in `data/output/{image_name}/` with:
- Corrected perspective image
- Detected board image
- Segmented hexagons
- Final board visualization
- JSON representation of the board state

