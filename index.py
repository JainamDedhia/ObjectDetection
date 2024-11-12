from ultralytics import YOLO

# Load a custom-trained YOLOv11 model (pre-trained on COCO dataset)
model = YOLO("yolo11x.pt")  # Use your custom YOLOv11 model file

# Train the model on the COCO8 dataset (or your custom dataset) for 100 epochs with a batch size of 16
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16)

# Run inference with the trained YOLOv11 model on a sample image ('bus.jpg')
results = model("path/to/bus.jpg")

# Print or visualize the results of the inference (optional)
results.show()  # Shows the image with predictions