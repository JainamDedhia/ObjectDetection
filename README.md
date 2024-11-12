# ObjectDetection
# YOLOv11 Custom Object Detection

This repository contains code for training a custom YOLOv11 object detection model using the COCO8 dataset (or any custom dataset). The project includes training instructions, model details, and inference capabilities for detecting objects in images.

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Training the Model](#training-the-model)
4. [Inference](#inference)
5. [Results](#results)

## Overview

The goal of this project is to train a custom YOLOv11 object detection model that can detect various objects. We have trained the model on the COCO8 dataset (a subset of the popular COCO dataset) and achieved varying levels of performance across three models. Two of the trained models performed well, while the third model showed decent results.

### YOLOv11 Model Details
- **Architecture**: YOLOv11 (a variant of the YOLO series, optimized for faster and more accurate object detection)
- **Dataset**: COCO8 (or your custom dataset)
- **Training Duration**: 100 epochs
- **Image Size**: 640x640
- **Batch Size**: 16

## Installation

To use this repository, you'll need Python and a few dependencies. Follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/YOLOv11-object-detection.git
   cd YOLOv11-object-detection
   ```

2. Install the required dependencies. You can install them manually with the following commands:

   ```bash
   pip install torch torchvision ultralytics
   ```

   These are the main dependencies required for training and running YOLOv11:
   - `torch`: The PyTorch framework for deep learning.
   - `torchvision`: Provides computer vision datasets, transformations, and models.
   - `ultralytics`: The library that provides the YOLOv11 model.

3. Download the pre-trained YOLOv11 model (or use your custom `.pt` file) and place it in the project directory.

## Training the Model

1. Prepare your dataset in YOLO format (or use the COCO8 dataset). Make sure the dataset is organized with images and a corresponding annotation file.

2. Configure the dataset YAML file (`coco8.yaml` in this case), which defines the dataset paths and class names.

3. Run the training script:
   ```python
   from ultralytics import YOLO

   # Load a custom-trained YOLOv11 model (pre-trained on COCO dataset)
   model = YOLO("yolo11x.pt")  # Use your custom YOLOv11 model file

   # Train the model on the COCO8 dataset (or your custom dataset) for 100 epochs with a batch size of 16
   results = model.train(data="coco8.yaml", epochs=100, imgsz=640, batch=16)
   ```

   - The training will run for 100 epochs, saving the best models after each epoch.

4. Once training is complete, the model weights will be saved in the `runs/` directory.

## Inference

After training, you can run inference on new images using the trained model:

```python
from ultralytics import YOLO

# Load the trained YOLOv11 model
model = YOLO("path/to/best_model.pt")  # Use the best-performing model

# Run inference with the trained model on a sample image
results = model("path/to/bus.jpg")

# Print or visualize the results of the inference (optional)
results.show()  # Shows the image with predictions
```

- Replace `path/to/best_model.pt` with the path to the model weights you want to use.
- Replace `path/to/bus.jpg` with the path to the image you want to process.

## Results

### Model 1: Excellent Results
- The first model achieved **high accuracy** on the validation set, showing strong performance for detecting objects.
 [Watch the car video](https://github.com/JainamDedhia/ObjectDetection/blob/main/car.avi)


### Model 2: Good Results
- The second model showed **solid accuracy** with a slightly lower performance compared to Model 1, but still sufficient for most tasks.

### Model 3: Decent Results
- The third model showed **acceptable results**, with some inconsistencies in detecting smaller objects or more complex scenes.
