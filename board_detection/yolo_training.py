from ultralytics import YOLO

yolo = YOLO('yolov8n.pt')
yolo.train(data='data/full/perspective_distorted_boards/data.yaml', epochs=100)
valid_results = yolo.val()
