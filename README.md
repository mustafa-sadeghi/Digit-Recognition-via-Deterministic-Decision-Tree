

# Classical Digit Recognition via Deterministic Decision Tree

A fully rule-based digit recognition pipeline using classical Computer Vision techniques.  
This project classifies digits through a deterministic decision tree built on geometric and morphological rules—without any machine learning.

---

## Description

A deterministic, rule-based digit recognition pipeline using classical Computer Vision.  
It applies thresholding, contour detection, ROI normalization, and classifies digits through geometric and morphological rules.  
The final output includes annotated bounding boxes.

---

## Features

- Deterministic and interpretable decision-tree classifier  
- Otsu thresholding and contour-based digit extraction  
- Height-normalized ROIs for consistent analysis  
- Rule-based nodes: hole detection, polygon approximation, tall-stroke analysis, centroid checks  
- Complete debug logs saved in `debug_steps/`  
- Final annotated result saved as an image

---

## Project Structure

```text
.
├── main.py                 # Main implementation (DigitClassifier)
├── Im321.png               # Input sample image
├── debug_steps/            # Generated intermediate outputs
│   ├── overview/
│   └── rois/
└── MV5_Sadeghi.pdf         # Full project documentation
```

## How It Works
1. Convert image to grayscale  
2. Apply Otsu thresholding  
3. Extract and filter contours  
4. Normalize each ROI  
5. Classify using decision-tree rules  
6. Save all intermediate steps  
7. Generate final annotated output

---

## Installation
```bash
pip install opencv-python numpy matplotlib
```
## Run the Project
```bash
python main.py
```

## Documentation
Full algorithm description, design rationale, and parameter tables are available in:
MV5_Sadeghi.pdf
