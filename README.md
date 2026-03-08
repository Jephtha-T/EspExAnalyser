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

## Repository Structure
```
EspExAnalyser/
├── Frame_Extraction.py              # Extract frames from video
├── Portafilter_Detection.py         # Detect portafilter ROI
├── Portafilter_Tracking.py          # Track and stabilise portafilter
├── Feature_Extraction.py            # Extract blonding and channeling features
├── Data_Export.py                   # Export analysis data to CSV for AI training
├── Espresso_Analysis.py             # Main pipeline orchestrator
├── README.md                        # Main ReadMe
│
├── Video Data/                      # Input videos
│   ├── test1.mp4
│   ├── test2.mp4
│   ├── labels.csv                   # Video metadata and quality labels (input)
│   └── ...
│
├── Image Data/
│   ├── Frames/                      # Extracted frames from video
│   └── Cropped/                     # Cropped and stabilised portafilter ROI
│
└── Analysis/                        # Output analysis results
    ├── Blond.png            # Blonding curve plot
    ├── Channeling.png       # Channeling detection plot
    ├── results.json         # Analysis metrics (frame #, rate, etc.)
    └── training_data.csv            # Exported features for AI model (output)
```

### Pipeline Stages

```
Video Input
    ↓
[1] Frame Extraction
    └─→ Frames/ (extracted frames)
    ↓
[2] Portafilter Tracking (Detection + Cropping + Stabilisation)
    └─→ Cropped/ (stabilised cropped frames)
    ↓
[3] Feature Extraction (Blonding & Channeling)
    └─→ Analysis/ (results, plots, JSON)
```

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
## Run Complete Pipeline

```bash
python Espresso_Analysis.py --mode single --video-name test1.mp4
```

Run every video in `Video Data/`:
```bash
python Espresso_Analysis.py --mode all
```

This will:
1. Extract frames to `Image Data/Frames/`
2. Detect, crop, and stabilise to `Image Data/Cropped/`
3. Extract features and save results to `Image Data/Analysis/`

## Export Data for AI Training

After running the analysis pipeline on your videos, you can export the extracted features to CSV format for machine learning model training:

Set `EXPORT_MODE = True` in `Espresso_Analysis.py` (line 24) and run:
```bash
python Espresso_Analysis.py
```

This generates `Analysis/training_data.csv` with:
- **Raw features**: shot_time, blonding_rate, channeling_avg/max/min
- **Normalized features**: All numeric features scaled to [0, 1] using min-max normalization
- **Labels**: Extraction quality from `Video Data/labels.csv` (e.g., "Ideal", "Under-extracted", "Over-extracted")
Notes:
- Scripts currently use fixed local paths and constants at the top of each file.
- Parameters (thresholds, ROI settings, frame rate) are intentionally exposed in code for tuning during experimentation.

## Author
Jephtha Ashter Tandri (20600677)
