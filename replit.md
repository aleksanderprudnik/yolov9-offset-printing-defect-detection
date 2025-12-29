# replit.md

## Overview

YOLOv9 is a state-of-the-art object detection framework implementing the paper "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information". This repository provides a complete deep learning pipeline for training, validating, and deploying object detection models. The codebase supports multiple model variants (T, S, M, C, E) with varying parameter counts and performance characteristics, and includes specialized functionality for offset printing defect detection using tile-based image processing.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core Detection Framework
- **Model Architecture**: YOLOv9 implements GELAN (Generalized Efficient Layer Aggregation Network) with Programmable Gradient Information (PGI) for improved feature learning
- **Model Variants**: Support for single, dual, and triple detection heads through separate training scripts (`train.py`, `train_dual.py`, `train_triple.py`)
- **Detection Pipeline**: Standard YOLO detection flow with configurable confidence thresholds, NMS (Non-Maximum Suppression), and multi-backend inference support

### Training and Validation
- **Training Scripts**: Separate implementations for different architectures - standard detection, dual-head detection, and triple-head detection
- **Loss Functions**: Task-Aligned Loss (TAL) implementations in `utils/loss_tal.py`, `utils/loss_tal_dual.py`, and `utils/loss_tal_triple.py`
- **Validation**: Corresponding validation scripts for each training variant with COCO-compatible metric computation

### Multi-Task Support
- **Object Detection**: Primary use case with bounding box prediction
- **Instance Segmentation**: Full segmentation pipeline in `segment/` directory
- **Classification**: Image classification support in `classify/` directory
- **Panoptic Segmentation**: Combined instance and semantic segmentation in `panoptic/` directory

### Image Processing Pipeline
- **Tile-Based Processing**: Custom tile creation for high-resolution images using SAHI library
  - 256×256 pixel tiles with 20% overlap
  - Designed for offset printing defect detection workflow
- **Image Reconstruction**: Post-detection tile merging in `reconstraction.py` to restore full-resolution annotated images
- **Data Augmentation**: Albumentations integration for training-time augmentation

### Model Export and Deployment
- **Export Formats**: PyTorch, TorchScript, ONNX, ONNX End2End, OpenVINO, TensorRT, CoreML, TensorFlow (SavedModel, GraphDef, Lite, Edge TPU, JS), PaddlePaddle
- **Multi-Backend Inference**: `DetectMultiBackend` class supports inference across all exported formats
- **Device Support**: CPU and CUDA GPU inference with automatic device selection

### Utility Infrastructure
- **Data Loading**: Custom dataloaders supporting images, videos, streams, and webcam input
- **Metrics**: mAP calculation, confusion matrix, precision-recall curves
- **Logging**: TensorBoard integration with optional Comet ML and ClearML support
- **Callbacks**: Extensible hook system for training lifecycle events

## External Dependencies

### Deep Learning Framework
- **PyTorch** (≥1.7.0): Primary deep learning framework
- **TorchVision** (≥0.8.1): Image transformations and pretrained models

### Computer Vision
- **OpenCV** (opencv-python ≥4.1.1): Image processing and visualization
- **Pillow** (≥7.1.2): Image file handling
- **Albumentations** (≥1.0.3): Advanced image augmentation

### Scientific Computing
- **NumPy** (≥1.18.5): Numerical operations
- **SciPy** (≥1.4.1): Scientific computing utilities
- **Pandas** (≥1.1.4): Data manipulation and analysis

### Object Detection Utilities
- **SAHI** (0.11.18): Slicing Aided Hyper Inference for tile-based processing
- **pycocotools** (≥2.0): COCO dataset format handling and evaluation

### Visualization and Logging
- **Matplotlib** (≥3.2.2): Plotting and visualization
- **Seaborn** (≥0.11.0): Statistical visualization
- **TensorBoard** (≥2.4.1): Training metrics logging

### Model Analysis
- **thop** (≥0.1.1): FLOPs and parameter counting

### Dataset Configuration
- Datasets stored in YAML format (e.g., `data/coco.yaml`)
- Images expected in `data/images/` directory structure
- Labels in YOLO format (class_id, x_center, y_center, width, height)