from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")
yolo.train(data="board_detection/data/input/data.yaml", epochs=100)
valid_results = yolo.val()
