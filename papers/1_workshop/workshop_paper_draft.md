# Automated Offset Printing Defect Detection System Based on YOLOv9 with Tile-Based Image Processing

---

## 1. Introduction

### 1.1 Problem Motivation

Offset lithography remains one of the dominant printing technologies in commercial publishing, packaging, and industrial manufacturing. The global offset printing market was valued at approximately $3 billion in 2024, with projections indicating continued growth driven by demand in packaging and label printing sectors [1]. Within the broader commercial printing industry, which exceeds $530 billion annually, offset printing maintains a significant market share due to its cost-effectiveness for medium to high-volume production runs [2].

Print quality defects represent a persistent challenge in offset printing operations. Common defects include ink spots, streaking, registration errors, hickeys, and ghosting artifacts. These defects result in measurable economic losses through several mechanisms: rejected products require disposal or recycling, reprinting consumes additional materials and press time, and defective products reaching customers damage brand reputation and generate returns.

Traditional quality control in offset printing relies on manual visual inspection. Trained operators examine printed output at designated inspection stations, identifying defects through visual comparison against reference samples. This approach presents several limitations. First, inspection throughput is constrained by human attention capacity; an experienced inspector can reasonably examine several hundred sheets per hour, a rate insufficient for modern presses operating at 10,000 to 15,000 impressions per hour. Second, manual inspection introduces subjectivity; studies of visual inspection tasks in manufacturing report inter-inspector agreement rates below 80% for subtle defects. Third, inspector performance degrades over extended periods due to fatigue, with error rates increasing measurably after the first few hours of a shift.

The gap between press speeds and inspection capacity creates a sampling problem: only a fraction of output receives detailed examination. Defects occurring between sampling intervals may produce significant quantities of rejected material before detection. This limitation becomes particularly problematic in regulated industries such as pharmaceutical and food packaging, where quality standards require documented inspection of all output.

These constraints motivate the development of automated inspection systems capable of continuous, consistent defect detection at production speeds. Computer vision approaches offer the potential for real-time analysis of every printed sheet, eliminating sampling gaps and providing objective, repeatable quality assessment.

---

## References

[1] Research and Markets, "Offset Printing Market Size, Competitors & Forecast to 2030," Global Industry Analysts, 2024. Available: https://www.researchandmarkets.com/report/offset-printing

[2] Smithers, "The Future of Digital vs Offset Printing to 2024," Smithers Market Reports, 2024. Available: https://www.smithers.com/services/market-reports/printing/the-future-of-digital-vs-offset-printing-to-2024

---

*Draft in progress - Section 1.1 complete*
