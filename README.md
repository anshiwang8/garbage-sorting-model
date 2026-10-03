# Garbage Sorting

A real-time computer vision system that detects objects and classifies them as compost, garbage, recycling, or none.

This project was built as an introduction to machine learning and computer vision. It uses a MobileNetV2-based image classifier to categorize waste and a custom-trained YOLO object detector to isolate the object of interest from the camera background.

The project currently supports four classes:

- Compost
- Garbage
- None
- Recycling

`None` represents objects that should not be classified as regular garbage, compost, or recycling.

## Table of Contents

- [Background](#background)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Install](#install)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Known Issues](#known-issues)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

## Background

The goal of this project is to create a computer vision system that can receive live camera input and determine how an object should be disposed of.

The original model classified the entire camera frame using MobileNetV2. This caused background objects, people, and other visual information to influence predictions.

To reduce this problem, the application was expanded to use YOLO object detection before classification.

The current inference pipeline is:

```text
Camera
  ↓
YOLO object detector
  ↓
Detected object bounding box
  ↓
Crop detected object
  ↓
Resize to 224 × 224
  ↓
Convert BGR → RGB
  ↓
MobileNetV2 preprocessing
  ↓
MobileNetV2 classifier
  ↓
Compost / Garbage / None / Recycling
```

This separates the project into two machine-learning tasks:

1. **Object detection** — determine where the relevant object is.
2. **Image classification** — determine which waste class the detected object belongs to.

## Architecture

### Image classifier

The classifier uses **MobileNetV2** pretrained on ImageNet as a feature extractor.

The original ImageNet classification head is removed:

```python
MobileNetV2(
    weights="imagenet",
    include_top=False
)
```

The pretrained MobileNetV2 layers are initially frozen so that their existing visual features are preserved during classifier training.

The custom classification head consists of:

```text
MobileNetV2
  ↓
GlobalAveragePooling2D
  ↓
Dense(4, activation="softmax")
```

`GlobalAveragePooling2D` reduces MobileNetV2's final feature maps into a feature vector.

The `Dense(4)` layer produces probabilities for the four classes.

### Training configuration

The classifier uses:

- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Metric:** Accuracy

Sparse categorical crossentropy is used because the class labels are represented as integer values rather than one-hot vectors.

Example class mapping:

```text
0 → Compost
1 → Garbage
2 → None
3 → Recycling
```

Adam adjusts the trainable model weights according to the gradients calculated during training.

### Object detector

The application also uses **Ultralytics YOLO11 Nano**.

A pretrained YOLO11n model was fine-tuned on custom annotated waste images so that it focuses on waste objects instead of objects such as the person holding them.

During live inference, detections below a confidence threshold are rejected.

Current threshold:

```text
50%
```

If YOLO does not detect an object with sufficient confidence, the MobileNetV2 classifier does not attempt to classify the frame.

## Dataset

The dataset is separated into three groups:

```text
Training Dataset/
Validation Dataset/
Testing Dataset/
```

Each contains the four classes:

```text
Compost/
Garbage/
None/
Recycling/
```

Images are loaded using:

```python
tf.keras.utils.image_dataset_from_directory(...)
```

Keras:

- reads image files
- determines labels from directory names
- resizes images to `224 × 224`
- groups images into batches

MobileNetV2 preprocessing is then applied:

```python
dataset.map(
    lambda x, y: (preprocess_input(x), y)
)
```

The test and validation datasets remain separate from the images used to update model weights.

The dataset has gone through several expansions after real-world testing revealed class imbalance and missing examples.

## Model Training

The classifier is trained using Keras:

```python
history = full_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)
```

During training:

```text
Training image
  ↓
MobileNetV2 feature extraction
  ↓
Classification prediction
  ↓
Sparse categorical crossentropy
  ↓
Gradient calculation
  ↓
Adam weight update
```

An early version of the model reached:

```text
Training accuracy:   97.76%
Training loss:        0.0722

Validation accuracy: 95.93%
Validation loss:      0.1155
```

As additional and more difficult data was introduced, evaluation became more representative of real-world performance.

The trained classifier is saved as:

```text
models/trash_classifier.keras
```

YOLO training produces a custom detector such as:

```text
runs/detect/train-3/weights/best.pt
```

## Evaluation

The current test set reports an overall accuracy of approximately:

```text
83.96%
```

with an overall sparse categorical crossentropy loss of approximately:

```text
0.4573
```

Current per-class test performance:

| Class | Accuracy | Loss |
|---|---:|---:|
| Compost | 94.6% | 0.171 |
| Garbage | 56.2% | 1.215 |
| None | 89.5% | 0.303 |
| Recycling | 94.9% | 0.157 |

The main current classification problem is the **Garbage** class.

The confusion matrix shows that a large proportion of true Garbage images are incorrectly classified as `None`:

```text
True Garbage → Garbage: 675 / 1200 (56.2%)
True Garbage → None:    440 / 1200 (36.7%)
```

This suggests substantial visual or dataset overlap between the Garbage and None categories.

The project generates several diagnostic visualizations:

- test accuracy by batch
- test loss by batch
- test accuracy by class
- test loss by class
- confusion matrix

These metrics are used to identify class-specific weaknesses rather than relying only on overall accuracy.

## Install

Python 3.13 is currently used for this project.

Clone the repository and install the required libraries:

```bash
pip install tensorflow numpy opencv-python ultralytics matplotlib
```

The main technologies are:

- Python
- TensorFlow
- Keras
- MobileNetV2
- OpenCV
- NumPy
- Ultralytics YOLO
- Matplotlib

## Usage

### Train the classifier

Run:

```bash
python training.py
```

This:

1. loads the training and validation datasets
2. preprocesses images for MobileNetV2
3. loads the pretrained MobileNetV2 feature extractor
4. trains the custom classification head
5. evaluates validation performance
6. saves the trained classifier

### Test the classifier

Run:

```bash
python testing.py
```

The testing program loads the saved classifier and evaluates it against the untouched test dataset.

It can also generate diagnostic graphs and a confusion matrix.

### Train the YOLO detector

Run:

```bash
python yolotune.py
```

The YOLO dataset requires bounding-box annotations and a dataset YAML configuration.

The resulting custom detector is saved by Ultralytics as a `.pt` weights file.

### Run the live application

Run:

```bash
python application.py
```

The application:

1. opens the webcam using OpenCV
2. sends each frame through the YOLO detector
3. selects a confident detected object
4. crops the detected region
5. resizes the crop to `224 × 224`
6. converts BGR input to RGB
7. applies MobileNetV2 preprocessing
8. runs the classifier
9. displays the detected object and predicted class

Press:

```text
q
```

to close the live camera application.

## Project Structure

A simplified version of the project structure is:

```text
garbage sorting/
│
├── application.py
├── training.py
├── testing.py
├── yolotune.py
│
├── models/
│   └── trash_classifier.keras
│
├── graphs/
│   ├── accuracy.png
│   ├── loss.png
│   └── ...
│
├── data/
│   ├── Raw Data/
│   │   ├── Training Dataset/
│   │   │   ├── Compost/
│   │   │   ├── Garbage/
│   │   │   ├── None/
│   │   │   └── Recycling/
│   │   │
│   │   ├── Validation Dataset/
│   │   └── Testing Dataset/
│   │
│   └── yolodata/
│       ├── images/
│       ├── labels/
│       └── data.yaml
│
└── runs/
    └── detect/
        └── ...
```

## Known Issues

### Garbage / None confusion

The current largest source of error is confusion between the `Garbage` and `None` classes.

Approximately 36.7% of true Garbage examples in the current test set are classified as None.

Future work includes:

- reviewing misclassified Garbage images
- checking Garbage and None labels for overlap
- increasing the visual diversity of Garbage examples
- improving examples of tissues, contaminated paper, wrappers, and mixed household waste
- analyzing prediction confidence on misclassified images

### Dataset bias

Earlier versions of the dataset contained significantly more Recycling images than Garbage and Compost images.

Additional data has since been collected to reduce this imbalance, but dataset quality and class coverage remain important areas of development.

### Object localization

Generic pretrained YOLO initially detected the user rather than the waste object being held.

A custom YOLO model was therefore trained on annotated waste images.

Detection quality can still affect classification because an inaccurate bounding box may include excessive background information or crop out part of the object.

### Real-world generalization

High validation accuracy does not always translate directly to webcam performance.

Differences in:

- lighting
- camera quality
- object orientation
- backgrounds
- object condition
- partial occlusion

can affect predictions.

## Data Sources

The dataset combines images from multiple publicly available waste and object-image datasets.

Sources used during development include:

1. TrashNet  
   https://github.com/garythung/trashnet

2. Garbage Classification V2  
   https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2

3. Garbage Dataset Classification  
   https://www.kaggle.com/datasets/zlatan599/garbage-dataset-classification

4. Fresh and Spoiled Food Image Dataset  
   https://www.kaggle.com/datasets/maheen00shahid/fresh-and-spoiled-food-image-dataset

5. RealWaste  
   https://github.com/sam-single/realwaste

6. Mendeley Data  
   https://data.mendeley.com/datasets/fv28xxn4f3/3

7. Mendeley Data  
   https://data.mendeley.com/datasets/96g5pgfnfw/1

8. Mendeley Data  
   https://data.mendeley.com/datasets/w68f9w6jmm/1

9. Recyclable and Household Waste Classification  
   https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification

10. Waste Classification  
    https://www.kaggle.com/datasets/phenomsg/waste-classification

11. Waste Classification Data  
    https://www.kaggle.com/datasets/techsash/waste-classification-data

12. Amazon Berkeley Objects Dataset  
    https://amazon-berkeley-objects.s3.amazonaws.com/index.html

13. AlphaTrash Dataset  
    https://github.com/Patipol-BKK/alphatrash-dataset

14. Mendeley Data  
    https://data.mendeley.com/datasets/6ps7gtp2wg/1

Dataset files are not necessarily redistributed with this repository. Users should review the license and usage terms of each original dataset before downloading or redistributing its images.

## Contributing

Contributions, bug reports, and suggestions are welcome.

If contributing training data:

- ensure images are correctly labelled
- avoid duplicates between training, validation, and test sets
- provide the original data source
- include licensing information where applicable
- avoid moving test images into the training dataset

For code changes, open an issue or submit a pull request.

## License

UNLICENSED — Copyright © 2026.

No license has currently been selected for this repository. Until a license is added, the source code should not be assumed to grant permission for reuse, modification, or redistribution.
