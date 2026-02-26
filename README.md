# EspExAnalyser

Espresso Shot Extraction Analysis Tool for COMP3025 (Individual Dissertation).

## Overview
This project builds a computer-vision pipeline that analyses espresso extraction videos, detects extraction anomalies, and estimates shot quality (under-extracted, balanced, over-extracted). The system is designed to work on ordinary smartphone recordings and provide objective feedback for espresso calibration and training.

The work combines:
- Image processing and feature extraction (OpenCV)
- Classical machine learning baselines (e.g., SVM, Random Forest)
- Planned hybrid/deep-learning extensions
- Planned app/chatbot-style recommendation interface

## Project Aim
Develop an app/web-based espresso analysis tool that can:
- Accept a video of an espresso shot
- Detect the portafilter basket ROI and extraction stream
- Extract temporal and visual extraction features
- Detect defects (e.g., channelling, spurting/spraying, premature blonding)
- Predict extraction quality and provide actionable recommendations

## Current Implementation Status
Based on the interim report, the current codebase includes:
- Video frame extraction with orientation handling
- Portafilter ROI detection pipeline
- Early feature extraction work (including colour/brightness trend analysis)
- Initial analysis plots and intermediate outputs

### Interim Progress Snapshot
- Dataset collected: 42 videos total
- Controlled recordings: 30
- Real-world cafe recordings: 12
- Basket detection success: 81% overall
- Controlled success: 83%
- Real-world success: 75%

## Repository Structure
- `Frame_Extraction.py`: Extracts frames from raw video (target sampling, orientation correction).
- `Portafilter_Detection.py`: Multi-stage ROI detection for basket localisation.
- `Portafilter_Tracking.py`: Basket tracking/cropping pipeline (intermediate stage support).
- `Feature_Extraction.py`: Ongoing feature and anomaly-related extraction logic.
- `Espresso_Analysis.py`: End-to-end style analysis script for extraction signals and outputs.
- `Video Data/`: Input extraction videos.
- `Image Data/Frames`: Extracted frames.
- `Image Data/Cropped`: ROI-cropped basket frames.
- `Image Data/Stabilised`: Stabilised/processed frame sequences.
- `Image Data/Analysis`: Analysis plots and generated outputs.

## Environment
Recommended environment:
- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Pillow
- scikit-learn

Install example:
```bash
pip install opencv-python numpy matplotlib pillow scikit-learn
```

## How To Run (Current Pipeline)
Run from the `EspExAnalyser` directory.

1. Extract frames from source video:
```bash
python Frame_Extraction.py
```

2. Detect basket ROI / run detection pipeline:
```bash
python Portafilter_Detection.py
```

3. Run analysis/feature pipeline:
```bash
python Espresso_Analysis.py
```

Notes:
- Some scripts currently use fixed local paths and constants at the top of each file.
- Parameters (thresholds, ROI settings, frame rate) are intentionally exposed in code for tuning during experimentation.

## Methodology Summary
The planned full pipeline follows four stages:
1. Video frame extraction
2. Portafilter basket detection (ROI)
3. Feature extraction and anomaly detection
4. Quality classification and recommendations

Planned extracted signals include:
- Stream/flow consistency and width dynamics
- Colour progression and blonding behaviour
- Spatial unevenness indicative of channelling
- Spraying/spurting events outside expected stream region

## Current Limitations
From interim findings:
- ROI detection is sensitive to extreme lighting and blur.
- Feature extraction is not yet complete across all target anomalies.
- Dataset size is still limited for robust model training.
- Ground-truth labels are primarily sensory/taste-based (limited instrumented TDS data).

## Planned Next Steps
- Expand dataset toward 100+ labelled videos.
- Improve robustness of ROI detection under difficult lighting.
- Complete anomaly detection and quality classification models.
- Implement and evaluate taste profile estimation.
- Build user-facing interface with recommendation/chatbot support.

## Academic Sources
This README is based on project materials in:
- `../Academic/Project Proposal.docx`
- `../Academic/Interim Report.docx`

## Author
Jephtha Ashter Tandri (20600677)
