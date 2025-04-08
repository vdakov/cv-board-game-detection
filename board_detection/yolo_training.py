from ultralytics import YOLO

# Load YOLO model
yolo = YOLO("yolov8n.pt")

# Start training with updated arguments
yolo.train(
    data="board_detection/data/input/data.yaml",
    epochs=2,
    project="board_detection/data/output",
    imgsz=640,  # Image size
    batch=16,  # Batch size
    lr0=0.01,  # Initial learning rate
    lrf=0.1,  # Final learning rate scaling factor
    augment=True,  # Enable augmentations
    degrees=10.0,  # Random rotation in degrees
    translate=0.2,  # Random translation (x, y)
    scale=0.5,  # Random scaling factor
    shear=10.0,  # Shear angle
    hsv_h=0.015,  # Random hue change
    hsv_s=0.7,  # Random saturation change
    hsv_v=0.4,  # Random value (brightness) change
    flipud=0.5,  # 50% chance of vertical flip
    fliplr=0.5,  # 50% chance of horizontal flip
    workers=8,  # Number of workers for data loading
    device="cuda",  # Use GPU
    save_period=10,  # Save every 10th epoch
)

# Validate the model
valid_results = yolo.val(project="board_detection/data/output")
