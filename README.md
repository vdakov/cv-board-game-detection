# cv-board-game-detection

A computer vision system that detects, segments, and classifies Catan game boards from image. The system identifies the board's position, correct for perspective distortion, segments individual hexagon tiles, and classifies them by resource type and number label. A visualizer and synthetic data miner are also implemented.

## Pipeline steps

1. **Homography Correction:** In a picture of a Catan Board is looked at from a certain perspective and correspondingly the board is distorted via some way. This distortion can be looked at as a combination of shears, translations and rotations. In other words, it can be expressed as a matrix $\mathbf H$, called a **homography matrix.** Correspondingly, this transformation can be inverted and a board can be looked at from some “default” perspective. In our case, this is top-down. A neural network predicts $\mathbf H$ and then $\mathbf H^{-1}$ is computed and applied to the image.

2. **Board Detection and Cropping:** To best-recognize the board, after correcting the perspective, we need to focus on it. We use YOLO object detection to find a bounding box over our board.

3. **Board/Hexagon Segmentation:** We then segment the board into separate tiles using a SAM model. We combine neighboring segments using DBScan, find all resulting contours, and filter them based on proportion-based heuristics of an expected proper Catan board (forms hexagons, proper size in relation to the imag, etc).

4. **Tile Classification:** We classify each hex tile separately, as a result of cropping the input image based on the found hexagonally-shaped contours. We use a CNN to recognize the resource type of the tile, and an OCR for the number labels. The two are separated using masks.
    
5. **Board Assembly:** Using the classification data for each tile, and the segmented hexes for their relative positions, we reconstruct the board into a game state, of a format suitable for both JSON and visualization.

## How to run

TODO: Below is only an outline. Go into more details and rewrite some parts.

As a first step, set up a virtual environment with all dependencies found on `requirements.txt`. Then, all models need to be downloaded and/or trained:
- Perspective correction: train homography net.
- Board detection: Yolo contained in repo.
- Board segmentation: SAM is automatically downloaded first time script is ran
- Tile classification: train classification networks and [download Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)

Each one goes to the relevant `models` folder.

Then, just run the `pipeline.py` script with an `img_path` argument for choosing an input image. The output, along with all intermediate steps, will be found on the `data/output` folder under the same name subfolder.
