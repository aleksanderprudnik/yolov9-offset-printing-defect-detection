# Scientific Paper Blueprint: Offset Printing Defect Detection Using YOLOv9

---

## Title (Proposed)

**"YOLOv9-Based Real-Time Detection of Offset Printing Defects Using Tile-Based Image Processing"**

Alternative titles:
- "Deep Learning Approach for Automated Quality Inspection in Offset Printing: A YOLOv9 Framework"
- "GELAN-Powered Defect Detection System for Industrial Offset Printing Quality Control"

---

## Abstract (Bullet-Point Skeleton)

- **Problem Statement**: Offset printing quality control remains labor-intensive; manual inspection is error-prone and cannot scale with production speeds
- **Proposed Solution**: An automated defect detection system leveraging YOLOv9 architecture with tile-based preprocessing for high-resolution print inspection
- **Methodology Summary**: Images divided into 256×256 overlapping tiles (20% overlap), processed by YOLOv9-S model, reconstructed with bounding box annotations
- **Key Results**: [To be populated with actual metrics from training - mAP, precision, recall from runs/train/yolov9-s8/]
- **Contributions**: 
  - Adaptation of YOLOv9 for single-class defect detection in printing domain
  - Tile-based processing pipeline enabling high-resolution image analysis
  - End-to-end system from raw images to annotated defect visualization

---

## 1. Introduction

### 1.1 Problem Motivation
- Offset printing is widely used in commercial publishing, packaging, and industrial applications
- Print defects (streaks, spots, registration errors, color inconsistencies) lead to product rejection and financial losses
- Traditional manual inspection:
  - Time-consuming and subjective
  - Prone to human fatigue and inconsistency
  - Cannot keep pace with high-speed printing production lines
- Need for automated, real-time quality inspection systems

### 1.2 Industrial Relevance of Offset Printing Defect Detection
- Global offset printing market size and growth projections
- Cost of defective prints in terms of material waste, reprinting, and customer dissatisfaction
- Increasing demand for zero-defect manufacturing in packaging and publishing industries
- Integration potential with Industry 4.0 smart manufacturing systems

### 1.3 Gap in Existing Solutions
- Limited publicly available datasets for offset printing defects
- Most existing solutions focus on:
  - Textile defect detection
  - Surface inspection (metal, wood)
  - General manufacturing defects
- Challenges specific to offset printing:
  - High-resolution images requiring specialized processing
  - Subtle defect patterns that blend with intended print patterns
  - Real-time processing requirements for production environments
- Lack of end-to-end systems that handle large-format images

### 1.4 Research Questions & Scientific Novelty
This work addresses three key research questions that justify scientific contribution beyond engineering integration:

#### RQ1: Tile-based Processing Optimization for Print Inspection
- **Question**: How does tile overlap ratio and tile size affect defect detection accuracy at image boundaries?
- **Scientific Contribution**: Formal analysis of boundary defect detection accuracy as function of overlap ratio (currently 20% - but is this optimal?)
- **Expected Innovation**: Development of overlap-aware detection loss function or boundary refinement network
- **Measurable Outcome**: Quantitative improvement in boundary defect detection compared to fixed 20% overlap

#### RQ2: Single-Class Detection under High False Positive Risk
- **Question**: How can we minimize false positives in single-class detection where negative examples (intended print patterns) visually resemble positives?
- **Scientific Contribution**: Novel negative sampling strategy and hard negative mining specifically for printing domain
- **Expected Innovation**: Custom loss function that explicitly addresses print pattern vs. defect disambiguation
- **Measurable Outcome**: Precision improvement and false positive reduction on print-like patterns

#### RQ3: Domain-Specific Feature Learning for Print Defects
- **Question**: Do generic object detection features (trained on COCO) optimally represent print defect characteristics?
- **Scientific Contribution**: Comparative analysis of pretrained vs. from-scratch training; feature visualization for defects vs. generic objects
- **Expected Innovation**: Domain-adaptive feature extraction modules or print-specific backbone modifications
- **Measurable Outcome**: Ablation study showing impact of domain-specific adaptations

### 1.5 Scientific Contribution Summary
Rather than simple library integration, this work contributes:

- **C1 (Methodological)**: Formalized tile-based high-resolution defect detection framework with theoretical analysis of boundary effects
- **C2 (Algorithmic)**: Optimized tile overlap strategy and boundary-aware post-processing for seamless defect localization
- **C3 (Empirical)**: Comprehensive benchmark of YOLOv9 variants for single-class industrial defect detection with domain-specific analysis
- **C4 (Dataset)**: First public/semi-public offset printing defect detection dataset with detailed defect taxonomy
- **C5 (Practical)**: End-to-end reproducible system addressing real production challenges (high-resolution, real-time, tile reconstruction)
- **C6 (Analytical)**: Analysis of why generic object detection models succeed/fail on printing domain (feature analysis, failure modes)

---

## 2. Related Work

### 2.1 Prior Approaches in Print Defect Detection
- Traditional image processing methods:
  - Threshold-based segmentation
  - Edge detection algorithms (Sobel, Canny)
  - Template matching techniques
  - Morphological operations
- Classical machine learning approaches:
  - Feature extraction (SIFT, SURF, HOG) with SVM/Random Forest classifiers
  - Texture analysis methods
- Limitations of traditional methods:
  - Sensitivity to lighting conditions
  - Difficulty handling varying print patterns
  - High false positive rates

### 2.2 Relevant Computer Vision / ML Methods
- Evolution of object detection architectures:
  - Two-stage detectors: R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN)
  - Single-stage detectors: YOLO series, SSD, RetinaNet
- YOLO architecture evolution:
  - YOLOv1-v8: Key innovations and improvements
  - YOLOv9: Introduction of GELAN (Generalized Efficient Layer Aggregation Network)
  - Programmable Gradient Information (PGI) concept
- Related industrial inspection applications:
  - PCB defect detection
  - Textile quality inspection
  - Surface anomaly detection
  - Package integrity verification
- Tile-based / sliding window approaches for high-resolution images:
  - SAHI (Slicing Aided Hyper Inference)
  - Patch-based detection strategies

---

## 3. Dataset & Domain Description

### 3.1 Nature of Offset Printing Defects
- Types of defects in offset printing:
  - Ink spots and smears
  - Streaking and banding
  - Registration misalignment
  - Ghosting effects
  - Hickeys (debris marks)
  - Blanket defects
- Visual characteristics of defects:
  - Size variation (from microscopic to visible spots)
  - Color contrast with background
  - Location patterns (random vs. systematic)
- Challenges in defect detection:
  - Distinguishing defects from intended print patterns
  - Handling transparent/semi-transparent overlays
  - Variable defect sizes and aspect ratios

### 3.2 Dataset Composition
- **Source**: Custom dataset of offset printing samples (referenced in `data/dataset.yaml`)
- **Class Definition**: Single class - "defect" (nc: 1)
- **Data Split**:
  - Training set: `data/images/train`
  - Validation set: `data/images/val`
  - Test set: `data/images/original_test_data`
- **Image Formats**: BMP, JPG, TIF (as indicated in preprocessing scripts)
- **Annotation Format**: YOLO format (class_id, x_center, y_center, width, height - normalized)
- **Dataset Statistics**: [To be populated - number of images, number of defect instances, size distribution]

### 3.3 Preprocessing Pipeline
- **Step 1: Image Acquisition**
  - Original high-resolution images placed in `data/images/original_test_data`
- **Step 2: Tile Generation** (`create_tiles_for_testdata.py`)
  - Tile dimensions: 256×256 pixels
  - Overlap ratio: 20% (horizontal and vertical)
  - Purpose: Enable processing of large images while maintaining detection accuracy
  - Library: SAHI (Slicing Aided Hyper Inference)
  - Output: Tiles saved to `data/images/test`
  - Naming convention: `{original_name}_tile_{index}.jpg`
- **Step 3: Label Preprocessing** (for training data)
  - Bounding box coordinate transformation for tiled images
  - Handling of defects spanning multiple tiles
- **Step 4: Data Augmentation** (during training):
  - HSV augmentation: H±0.015, S±0.7, V±0.4
  - Geometric: translation (0.1), scale (0.9), horizontal flip (0.5)
  - Mosaic augmentation (probability: 1.0)
  - MixUp augmentation (probability: 0.15)
  - Copy-paste augmentation (probability: 0.3)

---

## 4. Methodology

### 4.0 Novel Methodological Contributions (To Be Developed)

**To strengthen novelty beyond library integration, implement ONE or MORE of:**

1. **Boundary-Aware Loss Function**
   - Modified IoU loss that penalizes false positives/negatives differently for tiles at image edges
   - Implementation: Custom loss in `utils/loss.py`
   - Expected impact: Improve boundary defect detection by 3-5%

2. **Adaptive Tile Overlap Module**
   - Neural network that predicts optimal overlap ratio per image or region
   - Input: image content + defect confidence map
   - Output: adaptive overlap percentage for each tile
   - Implementation: Lightweight CNN branch predicting overlap factor

3. **Hard Negative Mining for Print Patterns**
   - Custom data loader that identifies and emphasizes "hard negatives" (print patterns that fool the model)
   - Mining strategy: Bootstrap approach collecting high-confidence false positives
   - Implementation: Modify `create_dataloader` to implement printing-specific hard negative sampling

4. **Domain Adaptation Module**
   - Lightweight domain discriminator to adapt features from COCO to printing
   - Adversarial training component that reduces domain shift
   - Implementation: Add Domain Adversarial Neural Network (DANN) losses

5. **Boundary Refinement Network**
   - Secondary lightweight CNN that post-processes predictions at tile boundaries
   - Input: overlapping regions + initial detections
   - Output: refined bounding boxes
   - Implementation: Separate refinement head with ~50k parameters

**Choose 1-2 of above based on computational budget. Each adds 5-15% effort but +30-40% novelty score.**

### 4.1 Model Architecture(s)
- **Primary Model**: YOLOv9-S (Small variant)
  - Configuration file: `models/detect/yolov9-s.yaml`
  - Pretrained weights initialization: `./yolov9-s`
- **GELAN Backbone** (Generalized Efficient Layer Aggregation Network):
  - Conv layers for initial downsampling
  - RepNCSPELAN4 blocks for feature extraction
  - ADown modules for dimensionality reduction
  - Multi-scale feature pyramid: P3/8, P4/16, P5/32
- **Detection Head**:
  - SPPELAN (Spatial Pyramid Pooling Enhanced Layer Aggregation)
  - Feature fusion via upsampling and concatenation
  - DDetect module for final predictions
- **Available Model Variants**:
  - GELAN-T (Tiny), GELAN-S (Small), GELAN-M (Medium), GELAN-C (Compact), GELAN-E (Extended)
  - YOLOv9-T, YOLOv9-S, YOLOv9-M, YOLOv9-C, YOLOv9-E
- **Key Architectural Components** (from `models/common.py`):
  - Conv: Convolution + BatchNorm + SiLU activation
  - RepNCSPELAN4: Reparameterized CSP-ELAN block
  - Concat: Multi-scale feature concatenation
  - SPPELAN: Enhanced spatial pyramid pooling

### 4.2 Feature Extraction
- Backbone feature extraction at multiple scales
- Feature pyramid network (FPN) for multi-scale detection
- Skip connections for preserving spatial information
- Progressive downsampling: 2× at each stage (P1/2 → P2/4 → P3/8 → P4/16 → P5/32)

### 4.3 Training Procedure
- **Optimizer**: SGD (Stochastic Gradient Descent)
  - Initial learning rate: 0.01
  - Final learning rate: 0.01 (linear decay to lr0 × lrf)
  - Momentum: 0.937
  - Weight decay: 0.0005
- **Learning Rate Schedule**: Linear decay
- **Warmup**:
  - Warmup epochs: 3.0
  - Warmup momentum: 0.8
  - Warmup bias learning rate: 0.1
- **Training Configuration**:
  - Epochs: 70
  - Batch size: 64
  - Input image size: 256×256
  - Workers: 8
  - Single GPU training (device: '0')
- **Early Stopping**: Patience of 100 epochs
- **Model Checkpointing**: Best and last weights saved

### 4.4 Hyperparameters
- **Loss Weights**:
  - Box loss gain: 7.5
  - Classification loss gain: 0.5
  - Objectness loss gain: 0.7
  - DFL (Distribution Focal Loss) gain: 1.5
- **Anchor Settings**:
  - IoU training threshold: 0.2
  - Anchor-multiple threshold: 5.0
- **Augmentation Parameters** (see Section 3.3)
- **Classification Weights**:
  - cls_pw (positive weight): 1.0
  - obj_pw (positive weight): 1.0
- **Focal Loss**: gamma = 0.0 (disabled)
- **Label Smoothing**: 0.0 (disabled)

### 4.5 Loss Functions
- **Composite Loss** (from `utils/loss.py`, `utils/loss_tal.py`):
  - **Bounding Box Loss**: IoU-based loss (CIoU/GIoU)
  - **Classification Loss**: BCE with Logits Loss
    - Option for Focal Loss with γ=1.5, α=0.25
    - Quality Focal Loss variant available
  - **Objectness Loss**: BCE with Logits Loss
  - **Distribution Focal Loss (DFL)**: For precise localization
- **Label Smoothing**: Configurable BCE target smoothing
- **Loss Balancing**: Multi-scale loss balancing [4.0, 1.0, 0.25, 0.06, 0.02] for P3-P7

---

## 5. Experiments

### 5.1 Experimental Setup
- **Hardware Configuration**:
  - Training: GPU (device '0')
  - Inference: CPU-capable for deployment flexibility
- **Software Environment**:
  - Framework: PyTorch
  - Dependencies: Listed in `requirements.txt`
- **Training Protocol**:
  - Single training run with fixed seed (0) for reproducibility
  - Validation performed each epoch
  - Best model selected based on fitness metric
- **Inference Pipeline** (`detect_dual.py`):
  - Input: Tiled images (256×256)
  - Confidence threshold: [configurable]
  - IoU threshold for NMS: [configurable]
  - Output: Detected tiles with bounding boxes, labels in TXT format, cropped defects

### 5.2 Evaluation Metrics
- **Primary Metrics**:
  - Mean Average Precision (mAP@0.5)
  - Mean Average Precision (mAP@0.5:0.95)
  - Precision
  - Recall
  - F1 Score
- **Detection Metrics**:
  - IoU (Intersection over Union) at various thresholds
  - Confidence scores for detected defects
- **Per-Class Metrics**: (Single class - 'defect')
  - True Positives, False Positives, False Negatives
- **Inference Performance**:
  - Inference time per tile (ms)
  - Throughput (tiles/second)
  - End-to-end pipeline latency
- **Visualization Outputs**:
  - Confusion matrix (`confusion_matrix.png`)
  - Precision-Recall curve (`PR_curve.png`)
  - F1 curve (`F1_curve.png`)
  - Precision curve (`P_curve.png`)
  - Recall curve (`R_curve.png`)
  - Training results plot (`results.png`)

### 5.3 Baselines & Ablation Studies (Novel Experimental Components)

#### Novel Experiments to Establish Scientific Contribution:

**Experiment A: Tile Overlap Optimization (Addresses RQ1)**
- Systematic evaluation of overlap ratios: 0%, 10%, 20%, 30%, 40%
- Metric: Boundary defect detection F1 score vs. overlap ratio
- Expected finding: Optimal overlap value and diminishing returns curve
- Novel contribution: First empirical characterization for printing domain

**Experiment B: Feature Domain Analysis (Addresses RQ3)**
- Compare training paradigms:
  - Pretrained COCO weights → fine-tuned (current)
  - From-scratch training on print defects only
  - Progressive domain adaptation
- Method: Feature visualization (t-SNE, activation maps) for defects vs. COCO objects
- Expected finding: Visualization showing feature divergence between domains
- Novel contribution: Evidence of domain-specific learning requirements

**Experiment C: False Positive Analysis on Print Patterns (Addresses RQ2)**
- Create "hard negative" set: high-confidence predictions on non-defect print patterns
- Evaluate detection under controlled variations:
  - Different printing inks/colors
  - Print pattern density (sparse vs. dense)
  - Texture complexity
- Expected finding: Pattern-specific false positive sources
- Novel contribution: Identifies disambiguation challenges specific to printing

#### Standard Baselines:
- **Model Variants**: YOLOv9-T/S/M/C/E, GELAN-T/S/M/C/E (to show why S was chosen)
- **External Detectors**: YOLOv8, Faster R-CNN, EfficientDet (if computing budget allows)
- **Classical Methods**: Threshold + morphology, edge detection baselines (to justify deep learning need)

#### Critical Ablation Studies:
- Tile size: 128, 256, 512 pixels (impact on detection accuracy vs. inference time)
- Overlap ratio: 0%, 10%, 20%, 30%, 40% (boundary detection accuracy)
- Augmentation combinations: mosaic, mixup, HSV → identify essential augmentations for printing
- Pretrained vs. scratch: quantify domain transfer benefit
- Loss function variants: test focal loss, QFLoss impact on single-class detection

---

## 6. Results & Discussion

### 6.1 Quantitative Results
- **Training Performance** (from `runs/train/yolov9-s8/results.csv`):
  - Final training loss values
  - Validation mAP progression over epochs
  - Best epoch identification
- **Detection Performance**:
  - mAP@0.5: [To be extracted from training logs]
  - mAP@0.5:0.95: [To be extracted from training logs]
  - Precision at optimal threshold
  - Recall at optimal threshold
  - F1 Score
- **Inference Speed**:
  - GPU inference time
  - CPU inference time (for deployment scenario)
- **Model Efficiency**:
  - Model size (parameters count)
  - FLOPs comparison with other variants
- **Detection Samples**: Located in `runs/detect/yolov9_m_c__detect/`
  - Sample detections with bounding boxes
  - Cropped defect regions in `crops/defect/`

### 6.2 Qualitative Observations
- **Defect Localization Accuracy**:
  - Precision of bounding box placement
  - Handling of small vs. large defects
- **Challenging Cases**:
  - Defects at tile boundaries (overlap handling)
  - Low-contrast defects
  - Multiple overlapping defects
- **Reconstruction Quality** (`runs/restored/`):
  - Seamless tile merging
  - Bounding box continuity across tiles
- **Visual Inspection of Results**:
  - Sample images: `1_reconstructed.jpg`, `2_reconstructed.jpg`

### 6.3 Scientific Novelty Results (Key Analysis for Novelty Claims)

#### Analysis of Research Questions:
**RQ1 Results: Tile Overlap Impact**
- Plot: Detection F1 vs. overlap ratio (0%-40%)
- Finding: [Where does optimal overlap fall? 20% or elsewhere?]
- Scientific value: First quantitative study of overlap for printing defects
- Implication: Can be generalized to other high-resolution inspection tasks

**RQ2 Results: False Positive Sources**
- Confusion matrix between: defects, print patterns, artifacts
- Analysis: Which print pattern types cause false positives?
- Finding: Domain-specific negative samples that COCO models struggle with
- Scientific value: Identifies limitation of generic object detectors for printing

**RQ3 Results: Domain Transfer Effectiveness**
- Comparison: COCO-pretrained vs. from-scratch on printing task
- Feature visualization: t-SNE clusters showing feature space separation
- Finding: How much does COCO transfer hurt/help?
- Scientific value: Evidence for or against domain-specific training

#### Ablation Study Results:
- **Tile Size Impact**: Trade-off curve (small tiles → more predictions but higher overlap cost)
- **Augmentation Necessity**: Which augmentations are critical for printing domain?
- **Loss Function Variants**: Does focal loss help single-class detection?

### 6.4 Error Analysis
- **False Positives**:
  - Patterns misidentified as defects (distinguish from false positives due to overlap artifacts)
  - Domain-specific patterns (print texture, color transitions) that trigger false alarms
- **False Negatives**:
  - Missed small defects (size threshold analysis)
  - Low-confidence detections below threshold (confidence distribution analysis)
  - Defects at tile boundaries (boundary effect quantification)
- **Boundary Effects** (Novel Analysis):
  - Defects split across tiles: detection rate vs. overlap ratio
  - Overlap ratio adequacy: quantitative boundary detection accuracy
  - Reconstruction artifacts: visual analysis of tile merging quality
- **Confusion Matrix Analysis**: Interpretation of `confusion_matrix.png`
- **Domain Mismatch Failures**: Cases where model fails due to COCO pretrain limitations

---

## 7. Limitations & Future Work

### 7.1 Current Limitations
- **Dataset Scope**:
  - Single defect class (no defect type classification)
  - Limited to specific offset printing scenarios
  - Dataset size constraints
- **Processing Pipeline**:
  - Manual multi-step execution (not fully automated)
  - Fixed tile size may not be optimal for all defect scales
- **Real-Time Constraints**:
  - Current pipeline not optimized for real-time production line integration
  - CPU inference may be slow for high-throughput requirements
- **Generalization**:
  - Model trained on specific printing materials/patterns
  - May require retraining for different print types

### 7.2 Future Work
- **Multi-Class Defect Classification**: Extend to categorize defect types (spot, streak, registration, etc.)
- **Adaptive Tiling**: Dynamic tile size selection based on image content
- **Real-Time Optimization**:
  - Model quantization (INT8, FP16)
  - TensorRT/ONNX deployment optimization
  - Edge device deployment (Jetson, Intel NCS)
- **Semi-Supervised Learning**: Leverage unlabeled print samples
- **Active Learning**: Iterative model improvement with human-in-the-loop
- **Segmentation Extension**: Instance segmentation for precise defect boundaries (using existing segment models)
- **Dataset Expansion**: Larger, more diverse printing defect dataset with public release
- **Integration**: Camera-based real-time inspection system for production lines

---

## 8. Conclusion

### Summary Points
- Presented YOLOv9-based system for offset printing defect detection
- Demonstrated effectiveness of tile-based preprocessing for high-resolution images
- Achieved [mAP value] on custom printing defect dataset
- Provided complete pipeline from image acquisition to defect visualization
- Model suitable for CPU deployment, enabling cost-effective industrial integration

### Impact Statement
- Potential to significantly reduce manual inspection labor in printing industry
- Contributes to quality assurance in packaging and publishing sectors
- Framework extensible to other industrial visual inspection tasks

---

## References (Placeholder Section)

### Object Detection & YOLO
1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
2. Wang, C.-Y., et al. "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information." arXiv 2024.
3. Wang, C.-Y., et al. "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors." CVPR 2023.

### Industrial Defect Detection
4. [Relevant papers on print/textile defect detection]
5. [Surface inspection in manufacturing]
6. [Deep learning for quality control]

### Technical Components
7. [SAHI: Slicing Aided Hyper Inference]
8. [Feature Pyramid Networks for Object Detection]
9. [Focal Loss for Dense Object Detection]

### Datasets & Benchmarks
10. Lin, T.-Y., et al. "Microsoft COCO: Common Objects in Context." ECCV 2014.
11. [Relevant industrial inspection datasets]

### Transfer Learning & Domain Adaptation
12. [Pre-training strategies for object detection]
13. [Domain adaptation for industrial applications]

---

## GUIDE: How to Add Scientific Novelty (Addressing Your Concern)

### Problem Statement
*"This looks like just gluing libraries together - YOLOv9 + SAHI + PyTorch = detection system. Where's the novelty?"*

### The Answer: Three Layers of Contribution

**Layer 1: Application Novelty** (What we have now)
- First system for offset printing defect detection
- Addresses domain-specific problem
- Not enough alone for top-tier venues

**Layer 2: Methodological Novelty** (What to add - Section 4.0)
- Custom loss functions optimized for printing patterns
- Domain adaptation strategies
- Tile boundary handling innovations
- Hard negative mining for print patterns
- **Impact**: Moves from engineering to research

**Layer 3: Empirical/Analytical Novelty** (What to analyze - Section 5.3 & 6.3)
- Systematic study of tile overlap effects (RQ1)
- Domain feature analysis showing COCO limitations (RQ3)
- False positive taxonomy for printing (RQ2)
- Comprehensive ablation studies
- **Impact**: Publishable insights beyond the application

### Quick Implementation Roadmap (by Effort)

**Option A: Minimum Novelty (1-2 weeks)**
- ✅ Add Research Questions (RQ1, RQ2, RQ3) → already in blueprint
- ✅ Design experiments addressing these RQs → already in blueprint
- Implement: Tile overlap ablation study (Section 5.3, Experiment A)
- Implement: Domain transfer comparison (Section 5.3, Experiment B)
- Result: Publishable in domain-specific venues

**Option B: Strong Novelty (3-4 weeks)**
- Do everything in Option A
- Implement: One of {Boundary-Aware Loss, Hard Negative Mining}
- Add: Feature visualization + domain analysis
- Result: Publishable in general ML/CV conferences

**Option C: Significant Novelty (5-6 weeks)**
- Do everything in Option A
- Implement: Two of {Adaptive Overlap Module, Domain Adaptation, Boundary Refinement}
- Add: Complete theoretical framework + extensive ablations
- Result: Publishable in top-tier venues (CVPR/ICCV quality)

### Where to Start
1. Run all experiments from Section 5.3 with current model (Experiments A, B, C)
2. Analyze results from Section 6.3 (RQ1, RQ2, RQ3 results)
3. This alone proves it's not "just libraries" - you have novel insights
4. Then pick 1-2 methodological contributions from Section 4.0 to implement

---

## Appendix (Optional Sections)

### A. Detailed Model Configurations
- Full YAML configuration files for all model variants
- Layer-by-layer architecture breakdown

### B. Training Curves
- Loss curves over epochs
- mAP progression plots
- Learning rate schedule visualization

### C. Additional Detection Samples
- Extended gallery of detection results
- Failure case analysis with visual examples

### D. Deployment Guide
- Step-by-step deployment instructions
- Hardware requirements
- Integration code samples

---

## Code Repository Structure Reference

```
├── data/
│   ├── dataset.yaml          # Dataset configuration (nc=1, names=['defect'])
│   ├── hyps/                  # Hyperparameter configurations
│   └── images/                # Image data (train, val, test, tiles)
├── models/
│   ├── detect/                # Detection model architectures (YOLOv9, GELAN variants)
│   ├── segment/               # Segmentation model architectures
│   └── common.py              # Shared model components
├── utils/
│   ├── loss.py                # Loss function implementations
│   ├── metrics.py             # Evaluation metrics
│   └── dataloaders.py         # Data loading utilities
├── runs/
│   ├── train/yolov9-s8/       # Training outputs (weights, metrics, plots)
│   ├── detect/                # Detection results
│   └── restored/              # Reconstructed images with annotations
├── train.py                   # Training script
├── detect_dual.py             # Detection script
├── create_tiles_for_testdata.py  # Tile generation preprocessing
└── reconstraction.py          # Image reconstruction post-processing
```

---

*Blueprint Version: 1.0*  
*Generated from codebase analysis*  
*Ready for expansion into full scientific manuscript*
