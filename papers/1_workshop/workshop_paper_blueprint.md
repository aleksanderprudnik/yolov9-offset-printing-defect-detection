# Workshop Paper Blueprint: Offset Printing Defect Detection Using YOLOv9
## (Based on Implemented System - Ready for Writing)

---

## Title

**"Automated Offset Printing Defect Detection System Based on YOLOv9 with Tile-Based Image Processing"**

Alternative titles:
- "YOLOv9-S for Industrial Offset Printing Quality Inspection: A Practical Implementation"
- "Deep Learning-Based Defect Detection in Offset Printing Using Tile Decomposition"

---

## Abstract

### Structure (to be written as prose):
- **Context**: Offset printing is a widely used industrial printing method where quality control remains largely manual
- **Problem**: Manual inspection is time-consuming, subjective, and cannot scale with high-speed production lines
- **Solution**: This paper presents an automated defect detection system based on YOLOv9-S architecture with tile-based preprocessing for high-resolution print images
- **Method**: Original images are divided into 256×256 pixel overlapping tiles (20% overlap), processed by a trained YOLOv9-S detector, and reconstructed with defect annotations
- **Results**: The system achieves **mAP@0.5 of 0.485**, **precision of 83.4%**, and **recall of 51.5%** on a custom offset printing defect dataset
- **Conclusion**: The proposed system demonstrates feasibility of deep learning for automated print quality inspection and provides a foundation for industrial deployment

---

## 1. Introduction

### 1.1 Problem Motivation
- Offset printing is widely used in commercial publishing, packaging, and industrial applications
- Annual global offset printing market exceeds $XX billion (cite industry report)
- Print defects lead to:
  - Product rejection and material waste
  - Reprinting costs
  - Customer dissatisfaction and reputation damage
- Traditional manual inspection limitations:
  - Time-consuming: trained inspectors can examine only X prints per hour
  - Subjective: inter-inspector agreement varies
  - Fatigue-prone: accuracy decreases over extended periods
  - Cannot match high-speed production lines (thousands of prints per hour)
- Growing need for automated, consistent quality inspection systems

### 1.2 Industrial Context of Offset Printing
- Offset printing process overview:
  - Ink transferred from plate to rubber blanket to printing surface
  - Multi-color printing requires precise registration
  - High-speed operation (10,000+ impressions per hour)
- Common defect types in offset printing:
  - Ink spots and smears
  - Streaking and banding
  - Registration misalignment
  - Ghosting effects
  - Hickeys (debris marks)
- Quality requirements:
  - Zero-defect standards in pharmaceutical and food packaging
  - Brand consistency requirements
  - Regulatory compliance in certain industries

### 1.3 Challenges in Automated Print Inspection
- High-resolution images required to detect small defects
- Defects must be distinguished from intended print patterns
- Real-time processing requirements for production integration
- Variable print designs require generalizable detection
- Limited availability of labeled training data

### 1.4 Paper Contributions
This paper presents a practical implementation with the following contributions:

- **C1**: Complete end-to-end system for offset printing defect detection using YOLOv9-S architecture
- **C2**: Tile-based preprocessing pipeline enabling processing of high-resolution print images
- **C3**: Image reconstruction methodology that combines tile-level detections into full-image visualization
- **C4**: Empirical evaluation on a custom offset printing defect dataset with detailed performance analysis
- **C5**: Open training configuration and reproducible experimental setup

### 1.5 Paper Organization
- Section 2: Related work in print defect detection and object detection methods
- Section 3: Dataset description and preprocessing pipeline
- Section 4: System architecture and methodology
- Section 5: Experimental setup and training configuration
- Section 6: Results and analysis
- Section 7: Discussion and limitations
- Section 8: Conclusion and future work

---

## 2. Related Work

### 2.1 Traditional Print Defect Detection Methods
- **Threshold-based approaches**:
  - Global and adaptive thresholding for defect segmentation
  - Limitations: sensitive to lighting, print density variations
- **Edge detection methods**:
  - Sobel, Canny operators for defect boundary detection
  - Limitations: difficulty distinguishing defects from print edges
- **Template matching**:
  - Comparison with reference "golden" sample
  - Limitations: requires perfect alignment, cannot handle design variations
- **Texture analysis**:
  - Gabor filters, Local Binary Patterns (LBP)
  - Limitations: domain-specific tuning required

### 2.2 Machine Learning Approaches
- **Classical ML with handcrafted features**:
  - SIFT, SURF, HOG feature extraction
  - SVM, Random Forest classifiers
  - Limitations: feature engineering burden, limited generalization
- **Early CNN approaches**:
  - AlexNet, VGG-based classification
  - Patch-based binary classification (defect/no-defect)

### 2.3 Modern Object Detection Architectures
- **Two-stage detectors**:
  - R-CNN, Fast R-CNN, Faster R-CNN
  - High accuracy but slower inference
- **Single-stage detectors**:
  - YOLO family: real-time detection with competitive accuracy
  - SSD, RetinaNet: multi-scale detection
- **YOLO evolution**:
  - YOLOv1-v3: foundational architectures
  - YOLOv4-v5: improved training strategies
  - YOLOv7: E-ELAN architecture innovations
  - YOLOv8: unified framework
  - **YOLOv9**: GELAN backbone, Programmable Gradient Information (PGI)

### 2.4 High-Resolution Image Processing
- **Sliding window approaches**:
  - Fixed-size window scanning
  - Computational overhead for large images
- **Tile-based processing**:
  - Image division into manageable tiles
  - Independent processing with result aggregation
- **SAHI (Slicing Aided Hyper Inference)**:
  - Systematic tile generation with overlap
  - Small object detection improvement

### 2.5 Related Industrial Inspection Applications
- PCB defect detection
- Textile quality inspection
- Surface anomaly detection (metal, wood)
- Packaging integrity verification

---

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

#### 3.1.1 Data Source
- Custom dataset of offset printing samples
- Images captured from production line / quality control samples
- (Describe acquisition conditions if known)

#### 3.1.2 Class Definition
- **Single class**: "defect"
- Binary detection task: defect present / absent
- Rationale for single-class approach:
  - Simplified labeling process
  - Primary goal is defect localization, not classification
  - Can be extended to multi-class in future work

#### 3.1.3 Dataset Statistics
| Property | Value |
|----------|-------|
| Number of classes | 1 ("defect") |
| Training images path | `data/images/train` |
| Validation images path | `data/images/val` |
| Test images path | `data/images/original_test_data` |
| Annotation format | YOLO format (class, x_center, y_center, width, height) |
| Image formats | BMP, JPG, TIF |

*[Minor Suggestion: Add exact counts of images and defect instances if available]*

#### 3.1.4 Defect Characteristics
- Size variation: from small spots to larger streaks
- Location: random distribution across print area
- Appearance: varying contrast against background

### 3.2 Tile-Based Preprocessing Pipeline

#### 3.2.1 Motivation for Tiling
- Original print images are high-resolution (dimensions exceed typical CNN input sizes)
- Direct resizing would lose small defect details
- Tiling preserves local detail while enabling batch processing

#### 3.2.2 Tiling Implementation
Using SAHI (Slicing Aided Hyper Inference) library:

```python
# From create_tiles_for_testdata.py
tile_size = 256  # pixels
overlap_ratio = 0.2  # 20% overlap

sliced_images = slice_image(
    image, 
    slice_height=256, 
    slice_width=256, 
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)
```

#### 3.2.3 Tiling Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Tile size | 256×256 pixels | Matches model input size, balances detail vs. context |
| Horizontal overlap | 20% (51 pixels) | Ensures defects at boundaries are fully captured |
| Vertical overlap | 20% (51 pixels) | Same as horizontal for consistency |
| Output format | JPEG | Efficient storage, acceptable quality loss |

#### 3.2.4 Tile Naming Convention
- Format: `{original_name}_tile_{index}.jpg`
- Example: `image1.bmp` → `image1_tile_0.jpg`, `image1_tile_1.jpg`, ...
- Enables reconstruction by parsing tile index

### 3.3 Data Augmentation (During Training)
Applied augmentations from hyperparameter configuration:

| Augmentation | Parameter | Value |
|--------------|-----------|-------|
| HSV-Hue | hsv_h | ±0.015 |
| HSV-Saturation | hsv_s | ±0.7 |
| HSV-Value | hsv_v | ±0.4 |
| Translation | translate | 0.1 |
| Scale | scale | 0.9 |
| Horizontal flip | fliplr | 0.5 (probability) |
| Mosaic | mosaic | 1.0 (probability) |
| MixUp | mixup | 0.15 (probability) |
| Copy-paste | copy_paste | 0.3 (probability) |

---

## 4. System Architecture and Methodology

### 4.1 Overall System Pipeline

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ Original Image  │───▶│ Tile         │───▶│ YOLOv9-S       │───▶│ Reconstruction   │
│ (High-res)      │    │ Generation   │    │ Detection      │    │ & Visualization  │
└─────────────────┘    └──────────────┘    └─────────────────┘    └──────────────────┘
     Input              Preprocessing         Inference            Post-processing
```

**Pipeline Steps:**
1. **Input**: High-resolution offset printing image (BMP/TIF/JPG)
2. **Tile Generation**: Divide into 256×256 tiles with 20% overlap
3. **Detection**: Process each tile through YOLOv9-S model
4. **Reconstruction**: Merge tile detections, draw bounding boxes on original image

### 4.2 Model Architecture: YOLOv9-S

#### 4.2.1 Architecture Overview
- **Model variant**: YOLOv9-S (Small)
- **Configuration file**: `models/detect/yolov9-s.yaml`
- **Backbone**: GELAN (Generalized Efficient Layer Aggregation Network)
- **Detection head**: Multi-scale detection at P3/8, P4/16, P5/32

#### 4.2.2 GELAN Backbone Components
| Component | Description |
|-----------|-------------|
| Conv | Standard convolution + BatchNorm + SiLU activation |
| RepNCSPELAN4 | Reparameterized CSP-ELAN block for feature extraction |
| ADown / AConv | Average pooling + convolution for downsampling |
| SPPELAN | Spatial Pyramid Pooling Enhanced Layer Aggregation |
| Concat | Multi-scale feature fusion |
| DDetect | Decoupled detection head |

#### 4.2.3 Feature Pyramid Structure
- **P3** (stride 8): Small object detection - 32×32 feature map for 256 input
- **P4** (stride 16): Medium object detection - 16×16 feature map
- **P5** (stride 32): Large object detection - 8×8 feature map

#### 4.2.4 Model Configuration
```yaml
# From yolov9-s.yaml
nc: 1  # number of classes (defect only)
depth_multiple: 1.0
width_multiple: 1.0
anchors: 3
```

### 4.3 Detection Process

#### 4.3.1 Inference Command
```bash
python detect_dual.py \
    --source 'data/images/test' \
    --img 256 \
    --weights 'runs/train/yolov9-s8/weights/best.pt' \
    --name yolov9_m_c__detect \
    --device cpu \
    --save-conf \
    --save-txt \
    --save-crop
```

#### 4.3.2 Detection Outputs
- **Annotated tiles**: Images with bounding boxes (`runs/detect/*/`)
- **Label files**: YOLO format detections (`runs/detect/*/labels/`)
  - Format: `class x_center y_center width height confidence`
- **Cropped defects**: Individual defect crops (`runs/detect/*/crops/defect/`)

### 4.4 Image Reconstruction

#### 4.4.1 Reconstruction Process
From `reconstraction.py`:
1. Parse tile filenames to identify original image
2. Calculate tile positions using index and overlap
3. Merge tiles into full-resolution image
4. Map detected bounding boxes to original image coordinates
5. Draw annotations on reconstructed image

#### 4.4.2 Reconstruction Parameters
```python
tile_size = 256
overlap = 0.2
tile_step = int(tile_size * (1 - overlap))  # 204 pixels
```

#### 4.4.3 Output
- Reconstructed images with defect bounding boxes
- Saved to: `runs/restored/`
- Example outputs: `1_reconstructed.jpg`, `2_reconstructed.jpg`

---

## 5. Experimental Setup

### 5.1 Training Configuration

#### 5.1.1 Hardware and Software
| Component | Specification |
|-----------|--------------|
| Training device | GPU (CUDA device 0) |
| Inference device | CPU (for deployment flexibility) |
| Framework | PyTorch |
| Model weights | Pretrained YOLOv9-S, fine-tuned on defect dataset |

#### 5.1.2 Training Hyperparameters
From `runs/train/yolov9-s8/opt.yaml` and `hyp.yaml`:

| Parameter | Value |
|-----------|-------|
| **Epochs** | 70 |
| **Batch size** | 64 |
| **Image size** | 256×256 |
| **Optimizer** | SGD |
| **Initial learning rate (lr0)** | 0.01 |
| **Final learning rate (lrf)** | 0.01 |
| **Momentum** | 0.937 |
| **Weight decay** | 0.0005 |
| **Warmup epochs** | 3.0 |
| **Warmup momentum** | 0.8 |
| **Warmup bias LR** | 0.1 |
| **Workers** | 8 |
| **Early stopping patience** | 100 |

#### 5.1.3 Loss Function Weights
| Loss Component | Weight |
|----------------|--------|
| Box loss | 7.5 |
| Classification loss | 0.5 |
| Objectness loss | 0.7 |
| DFL (Distribution Focal Loss) | 1.5 |

#### 5.1.4 Other Training Settings
| Setting | Value |
|---------|-------|
| IoU training threshold | 0.2 |
| Anchor-multiple threshold | 5.0 |
| Focal loss gamma | 0.0 (disabled) |
| Label smoothing | 0.0 (disabled) |
| Single class mode | False |
| Rectangular training | False |
| Multi-scale training | False |

### 5.2 Evaluation Metrics

#### 5.2.1 Primary Metrics
- **Precision**: TP / (TP + FP) - ratio of correct positive predictions
- **Recall**: TP / (TP + FN) - ratio of actual positives detected
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean AP averaged over IoU thresholds 0.5 to 0.95

#### 5.2.2 Loss Metrics (Training)
- **Box loss**: Bounding box regression loss
- **Classification loss**: Class prediction loss
- **DFL loss**: Distribution focal loss for precise localization

### 5.3 Available Evaluation Artifacts
Generated during training (`runs/train/yolov9-s8/`):

| File | Description |
|------|-------------|
| `confusion_matrix.png` | Classification confusion matrix |
| `F1_curve.png` | F1 score vs. confidence threshold |
| `P_curve.png` | Precision vs. confidence threshold |
| `R_curve.png` | Recall vs. confidence threshold |
| `PR_curve.png` | Precision-Recall curve |
| `results.png` | Training metrics over epochs |
| `results.csv` | Detailed epoch-by-epoch metrics |
| `labels.jpg` | Dataset label distribution |
| `labels_correlogram.jpg` | Label correlation analysis |

---

## 6. Results

### 6.1 Training Performance

#### 6.1.1 Final Model Performance (Best Epoch: 68)

| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.485** |
| **mAP@0.5:0.95** | **0.106** |
| **Precision** | **0.834** |
| **Recall** | **0.515** |

#### 6.1.2 Training Progression
| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| 10 | 0.214 | 0.204 | 0.096 | 0.020 |
| 20 | 0.427 | 0.331 | 0.259 | 0.056 |
| 30 | 0.564 | 0.364 | 0.298 | 0.047 |
| 40 | 0.708 | 0.420 | 0.364 | 0.079 |
| 50 | 0.650 | 0.385 | 0.365 | 0.073 |
| 60 | 0.800 | 0.485 | 0.466 | 0.102 |
| 68 (Best) | 0.834 | 0.515 | 0.485 | 0.106 |

#### 6.1.3 Training Loss Convergence
| Epoch | Box Loss | Cls Loss | DFL Loss |
|-------|----------|----------|----------|
| 1 | 1.871 | 33.25 | 1.219 |
| 10 | 4.330 | 3.517 | 1.671 |
| 30 | 3.549 | 2.262 | 1.442 |
| 50 | 3.347 | 2.001 | 1.390 |
| 68 | 3.274 | 1.905 | 1.361 |

### 6.2 Detection Examples

#### 6.2.1 Sample Detections
- Location: `runs/detect/yolov9_m_c__detect/`
- Detected tiles with bounding boxes for defects
- Cropped defect regions: `runs/detect/yolov9_m_c__detect/crops/defect/`

#### 6.2.2 Reconstructed Images
- Location: `runs/restored/`
- Examples: `1_reconstructed.jpg`, `2_reconstructed.jpg`
- Full-resolution images with detected defects annotated

### 6.3 Analysis

#### 6.3.1 Precision vs. Recall Trade-off
- High precision (83.4%): When model predicts defect, it's usually correct
- Moderate recall (51.5%): Model detects approximately half of actual defects
- Interpretation: Conservative detection - prefers false negatives over false positives
- Suitable for applications where false alarms are costly

#### 6.3.2 mAP Analysis
- mAP@0.5 (0.485): Moderate performance at standard IoU threshold
- mAP@0.5:0.95 (0.106): Lower performance at stricter IoU thresholds
- Indicates: Detection localization could be improved
- Bounding box precision is lower than classification accuracy

#### 6.3.3 Training Observations
- Model converged smoothly over 70 epochs
- Best performance achieved at epoch 68
- No significant overfitting observed (training loss continued decreasing)
- Early stopping not triggered (patience=100)

---

## 7. Discussion

### 7.1 Strengths of the Proposed System

1. **End-to-end solution**: Complete pipeline from raw images to annotated results
2. **High precision**: 83.4% precision minimizes false alarms
3. **Flexible deployment**: CPU inference enables cost-effective deployment
4. **Modular design**: Components can be upgraded independently
5. **Real-world applicability**: Tile-based approach handles high-resolution images

### 7.2 Limitations

#### 7.2.1 Dataset Limitations
- Single defect class (no defect type classification)
- Limited dataset size (specific to one printing scenario)
- Private dataset (not publicly available for benchmarking)

#### 7.2.2 Detection Limitations
- Moderate recall (51.5%): Misses approximately half of defects
- Lower mAP@0.5:0.95: Bounding box localization needs improvement
- Fixed tile size may not be optimal for all defect scales

#### 7.2.3 Pipeline Limitations
- Manual multi-step execution (not fully automated)
- No real-time processing optimization
- Tile boundary effects not explicitly handled

### 7.3 Comparison Context
- Direct comparison with other methods not performed (no public benchmark)
- Results demonstrate proof-of-concept feasibility
- Performance likely improvable with:
  - Larger, more diverse dataset
  - Hyperparameter optimization
  - Model architecture tuning

---

## 8. Conclusion and Future Work

### 8.1 Conclusion
This paper presented a practical deep learning system for automated offset printing defect detection. Key achievements:

- Successfully applied YOLOv9-S architecture to printing defect detection
- Developed tile-based preprocessing pipeline for high-resolution images
- Achieved 83.4% precision and 48.5% mAP@0.5 on custom dataset
- Demonstrated complete detection-to-visualization workflow
- Provided reproducible training configuration

The system demonstrates feasibility of deep learning for industrial print quality inspection and provides a foundation for production deployment.

### 8.2 Future Work

#### 8.2.1 Short-term Improvements
- Expand dataset with more defect examples and types
- Implement multi-class defect classification
- Optimize tile overlap ratio for boundary defect detection
- Add real-time inference optimization

#### 8.2.2 Medium-term Extensions
- Develop adaptive tiling based on image content
- Implement boundary-aware post-processing
- Deploy on edge devices (Jetson, Intel NCS)
- Integrate with production line camera systems

#### 8.2.3 Long-term Research Directions
- Domain adaptation for different printing types
- Semi-supervised learning with unlabeled production data
- Active learning for efficient dataset expansion
- Instance segmentation for precise defect boundaries

---

## References

[1] C.-Y. Wang, I.-H. Yeh, H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," arXiv preprint arXiv:2402.13616, 2024.

[2] J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016.

[3] F. C. Akyon, S. O. Altinuc, A. Temizel, "Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection," IEEE ICIP, 2022.

[4] (Add relevant offset printing / industrial inspection references)

[5] (Add relevant deep learning for manufacturing references)

---

## Appendix A: File Structure Reference

```
├── data/
│   ├── dataset.yaml                 # Dataset configuration (nc=1, names=['defect'])
│   ├── images/
│   │   ├── train/                   # Training images
│   │   ├── val/                     # Validation images
│   │   ├── original_test_data/      # Original high-res test images
│   │   └── test/                    # Generated tiles for testing
│   └── hyps/
│       └── hyp.scratch-high.yaml    # Hyperparameter template
├── models/detect/
│   └── yolov9-s.yaml                # Model architecture configuration
├── runs/
│   ├── train/yolov9-s8/
│   │   ├── weights/                 # Trained model weights (best.pt, last.pt)
│   │   ├── confusion_matrix.png     # [Figure 1]
│   │   ├── PR_curve.png             # [Figure 2]
│   │   ├── F1_curve.png             # [Figure 3]
│   │   ├── results.png              # [Figure 4]
│   │   └── results.csv              # Detailed training metrics
│   ├── detect/                      # Detection results
│   └── restored/                    # Reconstructed images
├── create_tiles_for_testdata.py     # Tile generation script
├── detect_dual.py                   # Detection script
├── reconstraction.py                # Image reconstruction script
└── train.py                         # Training script
```

---

## Appendix B: Figures List

| Figure | File | Description |
|--------|------|-------------|
| Figure 1 | `runs/train/yolov9-s8/confusion_matrix.png` | Confusion matrix |
| Figure 2 | `runs/train/yolov9-s8/PR_curve.png` | Precision-Recall curve |
| Figure 3 | `runs/train/yolov9-s8/F1_curve.png` | F1 score vs. threshold |
| Figure 4 | `runs/train/yolov9-s8/results.png` | Training metrics progression |
| Figure 5 | `runs/train/yolov9-s8/P_curve.png` | Precision vs. threshold |
| Figure 6 | `runs/train/yolov9-s8/R_curve.png` | Recall vs. threshold |
| Figure 7 | `runs/train/yolov9-s8/labels.jpg` | Dataset label distribution |
| Figure 8 | `runs/restored/1_reconstructed.jpg` | Sample detection result |

---

## Appendix C: Minor Improvement Suggestions (Low Effort)

These small additions would strengthen the paper without requiring significant extra work:

### C.1 Add Inference Speed Measurement
Run once and add to paper:
```bash
python detect_dual.py --source 'data/images/test' --img 256 --weights 'runs/train/yolov9-s8/weights/best.pt' --device cpu
# Note the inference time per image from output
```
**Adds**: Inference speed in ms/tile and throughput (tiles/second)

### C.2 Add Model Size Information
```python
import torch
model = torch.load('runs/train/yolov9-s8/weights/best.pt')
params = sum(p.numel() for p in model['model'].parameters())
print(f"Parameters: {params:,}")
```
**Adds**: Model parameters count, weight file size (MB)

### C.3 Add Dataset Statistics
Count images and annotations in train/val folders:
```bash
ls data/images/train | wc -l
ls data/images/val | wc -l
# Count annotation boxes in label files
```
**Adds**: Exact image counts, total defect instances, average defects per image

### C.4 Add Comparison Table (Optional)
If time permits, train YOLOv9-T and YOLOv9-M variants for comparison:
| Model | mAP@0.5 | Precision | Recall | Params | Speed |
|-------|---------|-----------|--------|--------|-------|
| YOLOv9-T | - | - | - | - | - |
| YOLOv9-S | 0.485 | 0.834 | 0.515 | - | - |
| YOLOv9-M | - | - | - | - | - |

---

*Workshop Paper Blueprint - Version 1.0*
*Describes implemented system only*
*Ready for expansion into full manuscript*
