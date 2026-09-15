# Age and Gender Detection with TensorFlow

A legacy computer-vision project for estimating **age** and **gender** from detected faces using a custom multi-output CNN built with TensorFlow/Keras and OpenCV.

## Project pipeline

```text
Camera / IP webcam
        ↓
OpenCV frame capture
        ↓
Haar Cascade face detection
        ↓
Face crop + resize to 48×48
        ↓
Pixel normalization (/255)
        ↓
CNN
   ┌────┴────┐
   ↓         ↓
 Gender     Age
```

## Repository contents

- `testing2.py` — original real-time inference script, updated only so it loads the available `models/Age_sex_detection.h5` file.
- `models/Age_sex_detection.h5` — complete saved Keras model.
- `models/trained.h5` — separately saved model weights.
- `haarcascade_frontalface_default.xml` — OpenCV Haar frontal-face detector.
- `legacy/model.ipynb` — original training/experimentation notebook.
- `legacy/model.json` — serialized Keras model architecture recovered from the original `json_file.txt`.

## Model

The recovered serialized architecture uses 48×48 RGB images and a CNN with convolutional blocks using 32, 64, 128 and 256 filters, followed by two 64-unit branches. The outputs are `sex_out` (sigmoid) and `age_out` (ReLU), in that order.

## Dataset workflow

The original notebook loads images from a UTKFace directory, extracts age and gender from the filenames, converts images to RGB, resizes them to 48×48, normalizes pixel values by 255, and creates train/test splits.

## Notes

This repository preserves the original project artifacts rather than rewriting the training notebook. Some paths and code reflect the original Google Colab/local development environment.

The original inference script referenced `New_Age_sex_detection.h5`; that file is not among the recovered artifacts, so the script now points to the available complete model at `models/Age_sex_detection.h5`.
