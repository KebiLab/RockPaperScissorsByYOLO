from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('runs/detect/train3/weights/best.pt')
#     results = model.train(data='data/data.yaml', epochs=20, imgsz=640, batch=8, device='0', optimizer='AdamW', lr0=0.001, degrees=15.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0)


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(data='data/data.yaml', epochs=50, imgsz=640, batch=8, device='0', optimizer='Adam', lr0=0.001, degrees=15.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0)


    # optimizer SGD Adam AdamW          optimizer='AdamW', lr0 = 0.001
    # lr0 0.01 0.001 0.001


