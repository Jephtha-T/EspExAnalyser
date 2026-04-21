# EspExAnalyser

Espresso shot analysis pipeline with a single-window Tkinter UI.

## Overview
This project analyses espresso extraction videos using classical computer vision:
- Portafilter detection and ROI review (auto + manual override)
- Stream/blonding analysis from colour and brightness changes
- Channeling estimation from FAST hole detections over time
- JSON output for each run and CSV export for dataset building

The current app is GUI-first (`Espresso_Analysis.py`) and supports single-video and batch processing.

## Current Pipeline
1. Frame extraction  
`Frame_Extraction.py` extracts frames at a target FPS, with a hard minimum of `1`.

2. Portafilter detection review  
`Portafilter_Detection.py` generates debug views and proposes a best basket ellipse.  
The UI shows 5 review images in one row:
- Original frame
- Pixel colour change mask
- FAST keypoints
- Detected features/edges
- Final chosen basket ellipse

3. Manual override (optional)  
User can draw a manual ellipse in the same app before full analysis.

4. Portafilter tracking/cropping  
`Portafilter_Tracking_v2.py` tracks the basket across the sequence and saves a locked ROI crop so the portafilter stays stable for downstream analysis.

5. Feature extraction  
`Feature_Extraction.py` computes:
- Blonding transition curve (from stream brightness progression)
- Channeling count per frame (holes detected by FAST)

6. Results UI + export  
`Espresso_Analysis.py` renders:
- Blonding chart
- Channeling chart
- Frame-sequence channeling preview
- Shot duration
- CSV export button (`Data_Export.py`)

## ROI Rules Used in Analysis
- Basket is represented as an ellipse.
- Analysis ROI is a rectangle from that ellipse bounding box.
- Rectangle height is extended to `2x` ellipse height (downward stream coverage).
- Both blonding and channeling calculations are restricted to this analysis ROI.

## Repository Structure
```text
EspExAnalyser/
|-- Espresso_Analysis.py       # Main Tkinter app (single window workflow)
|-- Frame_Extraction.py        # Video -> frames extraction
|-- Portafilter_Detection.py   # Basket ellipse detection + debug outputs
|-- Portafilter_Tracking.py    # Legacy fixed crop implementation
|-- Portafilter_Tracking_v2.py # Basket-tracked ROI locking for analysis
|-- Feature_Extraction.py      # Blonding/channeling feature extraction
|-- Data_Export.py             # Export analysis outputs to training CSV
|-- Espresso_Model.py          # Training/inference utilities for model experiments
|-- README.md
|-- Video Data/                # Input videos
|-- Image Data/
|   |-- Frames/                # Extracted frames
|   `-- Cropped/               # Cropped ROI frames
`-- Analysis/
    |-- results/               # Per-video results json
    `-- training_data.csv      # Exported dataset for model training
```

## Requirements
- Python 3.10+
- `opencv-python`
- `numpy`
- `matplotlib`
- `pillow`
- `scikit-learn`

Install:
```bash
pip install opencv-python numpy matplotlib pillow scikit-learn
```

## Run
From `EspExAnalyser/`:
```bash
python Espresso_Analysis.py
```

Then in the app:
- Choose `Single` or `Batch`
- For `Single`, browse a video file
- Review portafilter detection output
- Continue with auto ROI or draw manual ROI
- View final charts/preview and export if needed

## Outputs
- `Analysis/results/<video_name>_results.json`
- Optional dataset export from UI:
  - `Analysis/training_data.csv`

Note:
- Blonding/channeling plot images are not saved to disk in the current version.
- Legacy `extract_features_from_video(...)` args like `output_blond_dir`, `output_channeling_dir`, and `save_plots` are accepted for compatibility, but plots are handled in-app.

## Author
Jephtha Ashter Tandri (20600677)
