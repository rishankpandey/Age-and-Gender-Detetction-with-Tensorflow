# Age and Gender Detection with TensorFlow

A real-time computer vision application for estimating **age** and **gender** from detected faces using a custom multi-output CNN built with **TensorFlow/Keras** and **OpenCV**.

## Overview

The project combines OpenCV-based face detection with a convolutional neural network that processes detected faces and produces separate predictions for gender and age.

### Processing pipeline

```text
Camera / IP Webcam
        ↓
OpenCV Frame Capture
        ↓
Haar Cascade Face Detection
        ↓
Face Crop + Resize (48×48)
        ↓
Pixel Normalization (/255)
        ↓
Multi-Output CNN
     ┌───┴───┐
     ↓       ↓
  Gender    Age
```

## Features

- Real-time face detection using OpenCV Haar Cascades
- Face preprocessing and normalization
- Custom CNN for age and gender estimation
- Multi-output Keras model with separate prediction branches
- Support for camera/IP-webcam input
- Real-time prediction display using OpenCV

## Model Architecture

The model accepts **48×48 RGB images** and uses four convolutional blocks with:

- 32 filters
- 64 filters
- 128 filters
- 256 filters

The convolutional layers use 3×3 kernels, ReLU activations, max pooling, dropout, and L2 kernel regularization.

After feature extraction, the network branches into two 64-unit dense layers:

```text
Input: 48×48×3
      │
      ├── Conv2D (32) ── ReLU ── MaxPool ── Dropout
      ├── Conv2D (64) ── ReLU ── MaxPool ── Dropout
      ├── Conv2D (128) ─ ReLU ── MaxPool ── Dropout
      └── Conv2D (256) ─ ReLU ── MaxPool ── Dropout
                         │
                       Flatten
                         │
                 ┌───────┴───────┐
                 ↓               ↓
             Dense (64)       Dense (64)
                 ↓               ↓
             sex_out           age_out
             Sigmoid             ReLU
```

The model outputs are ordered as:

1. `sex_out` — gender prediction
2. `age_out` — age prediction

## Dataset Workflow

The training workflow uses the **UTKFace** dataset. Images are loaded from the dataset directory, with age and gender information extracted from filenames.

The preprocessing workflow includes:

1. Loading the images
2. Extracting age and gender labels
3. Converting images to RGB
4. Resizing images to 48×48 pixels
5. Normalizing pixel values by 255
6. Creating training and test splits

## Repository Structure

```text
Age-and-Gender-Detection-with-Tensorflow/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── testing2.py
├── haarcascade_frontalface_default.xml
│
├── models/
│   ├── Age_sex_detection.h5
│   └── trained.h5
│
└── legacy/
    ├── model.ipynb
    └── model.json
```

## Technologies

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Jupyter Notebook**
- **Haar Cascade Classifier**

## Running the Project

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The real-time inference script is:

```bash
python testing2.py
```

The application expects the following files to be available in the repository:

```text
models/Age_sex_detection.h5
haarcascade_frontalface_default.xml
```

## Model Files

Two H5 artifacts are included:

- `models/Age_sex_detection.h5` — complete saved Keras model
- `models/trained.h5` — separately saved model weights

The real-time inference script uses the complete `Age_sex_detection.h5` model.

## Notes

The project includes the training notebook and serialized model architecture alongside the trained model artifacts so that the complete training and inference workflow can be explored.

## License

This project is provided for educational and portfolio purposes.
