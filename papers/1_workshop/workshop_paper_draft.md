# Automated Offset Printing Defect Detection System Based on YOLOv9 with Tile-Based Image Processing

---

## 1. Introduction

### 1.1 Problem Motivation

Offset lithography remains one of the dominant printing technologies in commercial publishing, packaging, and industrial manufacturing. The global offset printing market was valued at approximately $3 billion in 2024, with projections indicating continued growth driven by demand in packaging and label printing sectors [1]. Within the broader commercial printing industry, which exceeds $530 billion annually, offset printing maintains a significant market share due to its cost-effectiveness for medium to high-volume production runs [2].

Print quality defects represent a persistent challenge in offset printing operations. Common defects include ink spots, streaking, registration errors, hickeys, and ghosting artifacts. These defects result in measurable economic losses through several mechanisms: rejected products require disposal or recycling, reprinting consumes additional materials and press time, and defective products reaching customers damage brand reputation and generate returns.

Traditional quality control in offset printing relies on manual visual inspection. Trained operators examine printed output at designated inspection stations, identifying defects through visual comparison against reference samples. This approach presents several limitations. First, inspection throughput is constrained by human attention capacity; an experienced inspector can reasonably examine several hundred sheets per hour, a rate insufficient for modern presses operating at 10,000 to 15,000 impressions per hour. Second, manual inspection introduces subjectivity; studies of visual inspection tasks in manufacturing report inter-inspector agreement rates below 80% for subtle defects. Third, inspector performance degrades over extended periods due to fatigue, with error rates increasing measurably after the first few hours of a shift.

The gap between press speeds and inspection capacity creates a sampling problem: only a fraction of output receives detailed examination. Defects occurring between sampling intervals may produce significant quantities of rejected material before detection. This limitation becomes particularly problematic in regulated industries such as pharmaceutical and food packaging, where quality standards require documented inspection of all output.

These constraints motivate the development of automated inspection systems capable of continuous, consistent defect detection at production speeds. Computer vision approaches offer the potential for real-time analysis of every printed sheet, eliminating sampling gaps and providing objective, repeatable quality assessment.

### 1.2 Industrial Context of Offset Printing

Offset lithography operates through an indirect transfer mechanism. Ink is first applied to an image-bearing plate, then transferred to an intermediate rubber blanket cylinder, and finally deposited onto the printing substrate. This indirect transfer provides consistent ink coverage and reduces plate wear, enabling long production runs. Modern offset presses achieve speeds exceeding 15,000 impressions per hour for sheet-fed configurations, with web-fed rotary presses reaching substantially higher throughput [3].

Multi-color printing compounds the complexity of quality control. A typical four-color (CMYK) press applies cyan, magenta, yellow, and black inks in sequence, with each color unit requiring precise registration to within fractions of a millimeter. Misregistration between color separations produces visible color fringing and blurred edges, particularly noticeable in text and fine graphic elements.

Defects in offset printing fall into several categories. Ink-related defects include spots and smears caused by contamination or improper ink-water balance, as well as streaking and banding resulting from uneven ink distribution across the roller train. Registration defects manifest as misalignment between color layers or between printed content and the substrate edge. Hickeys appear as small circular voids surrounded by ink halos, typically caused by debris particles on the blanket or plate. Ghosting produces faint duplicate images displaced from the primary image, resulting from ink starvation in high-coverage areas.

Quality requirements vary by application but trend toward increasingly stringent standards. Pharmaceutical packaging operates under regulatory frameworks that mandate zero-defect production with documented inspection records. Food packaging faces similar requirements, with additional constraints on ink migration and contamination. Commercial printing for brand owners demands consistent color reproduction across print runs and production sites, with tolerances defined by spectrophotometric measurement. These requirements establish the performance targets for any automated inspection system: high detection rates with minimal false positives, operating at full production speed.

### 1.3 Challenges in Automated Print Inspection

Automated visual inspection of printed materials presents several technical challenges that distinguish it from general object detection tasks.

Resolution requirements impose computational constraints. Print defects often measure less than one millimeter in diameter, while printed sheets may span several hundred millimeters. Capturing sufficient detail to detect small defects requires high-resolution imaging, producing images with dimensions that exceed typical neural network input sizes. Processing such images directly would demand prohibitive memory and computation; downsampling to manageable dimensions risks losing the fine detail necessary for defect detection.

Distinguishing defects from intended content represents a fundamental difficulty. Unlike industrial inspection of uniform surfaces, printed materials contain deliberate patterns, text, and graphics. A dark spot may be a defect or part of the design. An edge discontinuity may indicate a printing fault or the boundary of a graphic element. Detection algorithms must learn to identify anomalies within the context of varied and previously unseen print designs, rather than simply detecting deviations from a uniform background.

Production integration demands real-time performance. For inline inspection, processing latency must remain below the interval between successive sheets. At 15,000 impressions per hour, each sheet passes the inspection point approximately every 240 milliseconds. The complete pipeline—image acquisition, preprocessing, inference, and result handling—must complete within this window to avoid production bottlenecks.

Generalization across print designs poses additional challenges. A detection model trained on one set of printed materials may not transfer effectively to different designs, substrates, or ink formulations. Practical deployment requires either robust generalization or efficient adaptation to new production contexts.

Data availability constrains model development. Labeled datasets of printing defects are scarce in the public domain, as production data is typically proprietary and annotation requires domain expertise. This limitation affects both initial model training and the development of benchmark comparisons across methods.

### 1.4 Paper Contributions

This paper presents a practical implementation of automated defect detection for offset printing. The principal contributions are:

1. A complete end-to-end detection system based on the YOLOv9-S architecture, adapted for the printing defect domain.

2. A tile-based preprocessing pipeline that enables processing of high-resolution print images within standard deep learning frameworks.

3. An image reconstruction methodology that aggregates tile-level detections into full-resolution annotated output.

4. Empirical evaluation on a custom offset printing defect dataset, with quantitative performance analysis.

5. Documented training configuration and experimental setup to support reproducibility.

The following sections detail the dataset characteristics, system architecture, experimental methodology, and results.

## 2. Related Work

### 2.1 Traditional Print Defect Detection Methods

Early approaches to automated print inspection relied on classical image processing techniques. These methods remain relevant as baselines and are sometimes combined with modern deep learning approaches.

**Threshold-based segmentation** represents the simplest class of defect detection methods. Global thresholding separates foreground defects from background based on a fixed intensity value, while adaptive thresholding computes local thresholds to accommodate intensity gradients across the image [4]. These methods work adequately on uniform backgrounds but struggle with printed materials, where ink density varies by design and lighting conditions introduce additional intensity variation.

**Edge detection** using operators such as Sobel and Canny identifies discontinuities in image intensity that may correspond to defect boundaries [5]. The fundamental limitation for print inspection is that printed content contains numerous intentional edges—text characters, graphic borders, halftone patterns—that generate edge responses indistinguishable from defect boundaries without higher-level semantic understanding.

**Template matching** compares each printed sample against a reference "golden" image pixel by pixel or region by region. Difference maps highlight deviations that may indicate defects. This approach requires precise spatial registration between test and reference images; even sub-pixel misalignment produces false difference signals. Template methods also assume a fixed print design, requiring a new reference for each product changeover.

**Texture analysis** methods characterize local image patterns using filter banks such as Gabor wavelets or descriptors such as Local Binary Patterns (LBP). Anomalies appear as regions with texture statistics deviating from expected norms. While effective for detecting defects on textured surfaces in other manufacturing domains, these methods require careful parameter tuning for each printing application and may not generalize across different print designs or substrate materials.

### 2.2 Machine Learning Approaches

The transition from hand-crafted image processing to learned representations marked a significant shift in defect detection methodology.

**Classical machine learning** pipelines combined feature extraction with trainable classifiers. Feature descriptors such as Scale-Invariant Feature Transform (SIFT), Speeded-Up Robust Features (SURF), and Histogram of Oriented Gradients (HOG) provided compact representations of local image structure [6]. These features served as inputs to classifiers including Support Vector Machines and Random Forests, which learned decision boundaries separating defective from non-defective samples. The primary limitation of this approach lies in the feature engineering burden: designing discriminative features requires domain expertise and iterative refinement, and features optimized for one defect type or printing process may not transfer to others.

**Early convolutional neural network** approaches applied architectures such as AlexNet and VGG to industrial inspection tasks [7]. These networks learn hierarchical feature representations directly from image data, eliminating manual feature design. Initial applications to defect detection typically framed the problem as patch-based binary classification: images were divided into fixed-size patches, and each patch was classified as defective or non-defective. While this approach demonstrated improved accuracy over hand-crafted features, it provided only classification output without precise defect localization, and the patch-based formulation introduced boundary artifacts and computational redundancy.

### 2.3 Modern Object Detection Architectures

Object detection architectures provide both classification and localization in a single inference pass, making them suitable for defect detection tasks requiring spatial information.

**Two-stage detectors** separate region proposal from classification. The R-CNN family—R-CNN, Fast R-CNN, and Faster R-CNN—first generates candidate bounding boxes, then classifies each region independently [8]. Faster R-CNN introduced learnable Region Proposal Networks, achieving strong accuracy on benchmark datasets. However, the two-stage pipeline incurs computational overhead that limits real-time applicability in high-throughput industrial settings.

**Single-stage detectors** perform localization and classification simultaneously across the image. The YOLO (You Only Look Once) architecture treats detection as a regression problem, predicting bounding boxes and class probabilities directly from image features [9]. SSD (Single Shot Detector) introduced multi-scale feature maps for detecting objects at different sizes, while RetinaNet addressed class imbalance through focal loss. Single-stage detectors achieve inference speeds compatible with real-time requirements while maintaining competitive accuracy.

**The YOLO architecture** has evolved through multiple generations. YOLOv1 through YOLOv3 established the foundational single-stage paradigm with progressive improvements to accuracy and multi-scale detection. YOLOv4 and YOLOv5 introduced enhanced training strategies including mosaic augmentation, auto-anchor learning, and improved loss functions. YOLOv7 contributed the Extended Efficient Layer Aggregation Network (E-ELAN) for improved gradient flow. YOLOv8 unified detection, segmentation, and classification within a single framework.

**YOLOv9** represents the current state of the art, introducing two key innovations [10]. The Generalized Efficient Layer Aggregation Network (GELAN) backbone improves feature extraction through flexible computational block design. Programmable Gradient Information (PGI) addresses information loss during deep network training by providing auxiliary supervision paths. These contributions yield improved accuracy-efficiency trade-offs compared to prior YOLO versions. This work adopts YOLOv9-S (Small variant) as the detection backbone.

### 2.4 High-Resolution Image Processing

Processing high-resolution images with neural networks designed for fixed input dimensions requires strategies to preserve detail while managing computational constraints.

**Sliding window approaches** scan a fixed-size detection window across the image at multiple positions and scales. Each window position is processed independently through the detector, and overlapping detections are merged through non-maximum suppression. While straightforward to implement, sliding window methods incur substantial computational overhead: a high-resolution image may require hundreds or thousands of window evaluations, with significant redundant computation in overlapping regions.

**Tile-based processing** divides the input image into a grid of non-overlapping or overlapping tiles, processes each tile independently, and aggregates results into a unified output. This approach bounds memory consumption to that required for a single tile and enables parallel processing across tiles. The primary challenge lies in handling objects that span tile boundaries: without overlap, such objects may be truncated and missed by detection.

**SAHI (Slicing Aided Hyper Inference)** provides a systematic framework for tile-based object detection [11]. SAHI generates tiles with configurable size and overlap ratio, runs inference on each tile, maps detections back to original image coordinates, and merges overlapping predictions. The overlap between adjacent tiles ensures that objects near boundaries appear completely within at least one tile. SAHI has demonstrated particular effectiveness for small object detection, where objects occupy only a small fraction of high-resolution images and would be poorly resolved after downsampling to standard network input sizes. This work employs SAHI for tile generation in the preprocessing pipeline.

## 3. Dataset and Preprocessing

### 3.1 Dataset Description

#### 3.1.1 Data Source

The dataset comprises images captured from an operational offset printing production line using a custom quality control system. The acquisition infrastructure consists of a multi-threaded application orchestrating synchronized image capture from four Basler acA1440-73gm industrial cameras (1440×1080 resolution, monochrome, GigE interface) arranged to cover adjacent quadrants of the printed sheet. Camera synchronization with the printing cylinder is achieved through a hardware trigger system: a Hall-effect sensor mounted on the press cylinder generates trigger pulses, which are processed by a dedicated controller that calculates instantaneous press speed from inter-pulse intervals and coordinates acquisition timing across all camera units.

Each camera captures its designated region of interest with configurable margins to accommodate mechanical variability in sheet positioning (approximately ±1 mm). The acquisition system operates at production speed, capturing every printed sheet as it passes the inspection station. Images are stored in lossless format to preserve defect detail for subsequent annotation.

The original quality control system employed a template-matching approach for defect detection. Each camera maintained a grayscale reference template, and incoming frames were compared using the Structural Similarity Index (SSIM). Difference maps underwent Otsu thresholding followed by morphological filtering (erosion and dilation with 7×7 kernels, three iterations each) to suppress noise. Candidate defects were identified through contour analysis with a minimum area threshold of 250 pixels, and a temporal consistency filter required defects to persist across consecutive frames to distinguish genuine anomalies from transient artifacts. However, this approach exhibited fundamental limitations: the pixel-wise difference metrics produced false positives in the presence of mechanical sheet positioning variations, as such perturbations generated SSIM degradation patterns geometrically similar to actual sub-centimeter defects. These limitations motivated the transition to a learning-based detection approach.

#### 3.1.2 Class Definition

The dataset employs a single class label: "defect." This binary formulation (defect present/absent) simplifies the annotation task and focuses the detection objective on localization rather than fine-grained defect categorization. The single-class approach is appropriate for the current application, where the primary requirement is to flag any quality deviation for operator review. Extension to multi-class defect taxonomy (distinguishing ink spots, streaks, registration errors, etc.) represents a direction for future work contingent on expanded annotation effort.

#### 3.1.3 Dataset Statistics

| Property | Value |
|----------|-------|
| Number of classes | 1 ("defect") |
| Training images | Located in `data/images/train` |
| Validation images | Located in `data/images/val` |
| Test images | Located in `data/images/original_test_data` |
| Annotation format | YOLO format (class, x_center, y_center, width, height) |
| Supported image formats | BMP, JPG, TIF |

(Exact image counts and defect instance statistics to be added from dataset analysis.)

---

## References

[1] Research and Markets, "Offset Printing Market Size, Competitors & Forecast to 2030," Global Industry Analysts, 2024. Available: https://www.researchandmarkets.com/report/offset-printing

[2] Smithers, "The Future of Digital vs Offset Printing to 2024," Smithers Market Reports, 2024. Available: https://www.smithers.com/services/market-reports/printing/the-future-of-digital-vs-offset-printing-to-2024

[3] H. Kipphan, "Handbook of Print Media: Technologies and Production Methods," Springer, 2001.

[4] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," IEEE Transactions on Systems, Man, and Cybernetics, vol. 9, no. 1, pp. 62-66, 1979.

[5] J. Canny, "A Computational Approach to Edge Detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 8, no. 6, pp. 679-698, 1986.

[6] N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR, pp. 886-893, 2005.

[7] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," NeurIPS, pp. 1097-1105, 2012.

[8] S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS, pp. 91-99, 2015.

[9] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR, pp. 779-788, 2016.

[10] C.-Y. Wang, I.-H. Yeh, and H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," arXiv preprint arXiv:2402.13616, 2024.

[11] F. C. Akyon, S. O. Altinuc, and A. Temizel, "Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection," IEEE ICIP, pp. 966-970, 2022.

---

*Draft in progress - Sections 1, 2, 3.1 complete*
