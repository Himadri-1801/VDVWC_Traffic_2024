
---

# VDVWC_Competition_2024

This repository contains a YOLO-based object detection model for identifying various types of vehicles in an image. The model is trained on a dataset with diverse weather and time conditions, making it robust for different scenarios.

## Installation

To run the notebook in Google Colab, follow these steps:

1. Open the provided `.ipynb` file in Google Colab.
2. Run all cells sequentially.

The notebook performs the following tasks:
- Installs necessary package (`ultralytics`).
- Clones the dataset from GitHub.
- Prepares the dataset for training.
- Converts all `.jpeg` images to `.jpg`.
- Validates the dataset for inconsistencies.
- Trains the YOLO model on the provided dataset.
- Validates the trained model and calculates the Mean Average Precision (mAP).

## Dataset

The dataset is structured as follows:

```
juvdv2-vdvwc/
  ├── Train/
  │   ├── Rainny/
  │   │   ├── Day/
  │   │   └── Night/
  │   └── Sunny/
  │       ├── Day/
  │       └── Night/
  ├── Val/
  │   ├── Rainny/
  │   │   ├── Day/
  │   │   └── Night/
  │   └── Sunny/
  │       ├── Day/
  │       └── Night/
  └── Annotation/
      ├── Train/
      └── Val/
```

## Model Training

After running the training cells, the best model weights are saved at:
```
/content/runs/detect/train/weights/best.pt
```

## Evaluation Metrics

The model's performance on the validation dataset is as follows:
- Mean Average Precision @.5:.95 : 0.2818
- Mean Average Precision @ .50   : 0.5632
- Mean Average Precision @ .70   : 0.2550

## Prediction

To predict objects in an image using the best-trained model, use the following CLI command in your Colab environment:

```python
from ultralytics import YOLO

model = YOLO('/content/runs/detect/train/weights/best.pt')
results = model.predict(source='path/to/your/image.jpg')
```

Replace `'path/to/your/image.jpg'` with the path to the image you want to run predictions on.

## Visualization

The notebook also includes code to visualize some of the predictions on the validation dataset. It generates a grid of images with bounding boxes around detected objects.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request.

---
