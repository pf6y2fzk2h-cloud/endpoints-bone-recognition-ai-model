# endpoints-bone-recognition-ai-model
# Automated 2D:4D Digit Ratio Measurement from Hand X-rays

Master's thesis project — Umeå University - Laura Wagenbach 
A two-stage deep learning pipeline for automatically measuring the 2D:4D digit ratio 
from hand radiographs (X-rays).

## Overview

The 2D:4D digit ratio (index finger length / ring finger length) is a biomarker linked 
to prenatal androgen exposure. This project automates its measurement from hand X-rays 
using a cascade of two neural networks:

**Stage 1 — Finger Segmentation**  
A ResNet18-based U-Net segments the index (2nd) and ring (4th) fingers on both hands 
(L2, L4, R2, R4). Achieved a mean IoU of 85.60% across 220 images.

**Stage 2 — Keypoint Detection**  
A convolutional encoder-decoder detects 6 anatomical keypoints per finger (fingertip 
and base) within each segmented region. Finger lengths are computed as the Euclidean 
distance between keypoints, and the 2D:4D ratio is calculated from these lengths.

## Repository Structure

train_finger_seg.py      # Train the segmentation model (Stage 1)
train_finger_ep.py       # Train the keypoint detection model (Stage 2)
evaluate_seg.py          # Evaluate segmentation with IoU per finger class
evaluate.py              # Evaluate full pipeline (keypoint accuracy + 2D:4D ratio)
evaluate_test.py         # Visual evaluation on test images
predict_cascade.py       # Run full cascade prediction on new images
predict_xray_images.py   # Batch prediction on a folder of X-ray images
export_onnx.py           # Export trained models to ONNX format
model_summary.py         # Print model architecture summaries
manual_measurements.py   # Process manual measurement data
matched_comparison.py    # Compare AI vs. manual 2D:4D measurements

## Requirements

Python 3.12
torch
torchvision
numpy
opencv-python (cv2)
Pillow


Install with:
```bash
pip install torch torchvision numpy opencv-python Pillow


Usage:
Train:
python3.12 train_finger_seg.py   # Stage 1: segmentation
python3.12 train_finger_kp.py    # Stage 2: keypoint detection

Evaluate:
python3.12 evaluate_seg.py       # IoU per finger class
python3.12 evaluate.py           # Full pipeline evaluation

Predict:
python3.12 predict_cascade.py    # Run on test images


Data:
Training images and annotation files are not included in this repository due to
privacy constraints (medical imaging data). The expected input format is:

Grayscale hand X-ray images (.png or .jpg)
Polygon annotations in labels.json
Model Weights
Pre-trained model weights (.pth files) are not included due to file size.
Train from scratch using the training scripts above.

