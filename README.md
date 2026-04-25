# EspExAnalyser

EspExAnalyser is a desktop tool for analysing espresso extraction videos. It helps you review a shot visually, detect the portafilter area, track the extraction region, measure blonding and channeling behaviour, and export the results for later comparison or research.

This project is designed for users who want to run espresso shot analysis on recorded videos without having to work directly with the code.

## Features

- analyse a single espresso video or process all videos in a folder
- detect the portafilter automatically
- manually correct the detection when needed
- stabilise the basket region across frames for more consistent analysis
- measure blonding progression and channeling behaviour
- view plots and visual previews inside the app
- save per-video results as JSON
- export labeled CSV datasets for further study

## What You Need

- Windows with Python 3.10 or newer
- a folder of espresso extraction videos
- the required Python packages installed

Install the dependencies from inside the project folder:

```bash
pip install opencv-python numpy matplotlib pillow scikit-learn joblib pandas openpyxl ultralytics
```

Notes:
- `tkinter` is required for the desktop interface and is usually included with Python on Windows.
- If `ultralytics` is not installed, the app may still run, but the YOLO-based portafilter detection path will not be available.

## Project Layout

The folders you will use most are:

- `Video Data/` for input espresso videos
- `Image Data/Frames/` for extracted frames
- `Image Data/Cropped/` for stabilised cropped analysis frames
- `Analysis/results/` for saved JSON results
- `Analysis/` for exported CSV files

## How To Run

From the `EspExAnalyser` folder:

```bash
python Espresso_Analysis.py
```

When the app opens:

1. Choose `Single` if you want to analyse one video, or `Batch` to analyse all videos in `Video Data/`.
2. If using single mode, browse for the video you want.
3. Start the analysis.
4. Review the detected portafilter area.
5. Accept the result or draw a manual ellipse if the automatic result is off.
6. Wait for the analysis to finish and review the charts, previews, and summary information.
7. Export the results if needed.

## Recommended Workflow

For the smoothest experience:

1. Place your espresso videos in `Video Data/`.
2. Run the app in single mode first to confirm the detection works well on your videos.
3. Use manual correction on any shots where the basket is not detected cleanly.
4. Once the setup looks reliable, use batch mode for larger runs.
5. Export CSV data after you have collected enough labelled results.

## Output Files

Each analysed video saves a result file to:

```text
Analysis/results/<video_name>_results.json
```

The app can also export CSV files into `Analysis/`, including:

- `training_data.csv`
- `training_data_events.csv`
- `training_data_timeseries.csv`

These exports are useful if you want to compare many shots or build a labelled dataset from your results.

## Input Video Tips

Best results usually come from videos that:

- clearly show the bottom of the portafilter
- have stable camera framing
- keep the cup and stream visible throughout the shot
- avoid strong motion blur
- use consistent lighting

If the automatic detection is inaccurate, use the manual ellipse option in the app before continuing.

## Troubleshooting

If the app does not open:

- make sure Python 3.10+ is installed
- make sure all required packages were installed successfully
- check that `tkinter` is available in your Python installation

If the portafilter is not detected correctly:

- try a clearer frame or better-lit video
- use the manual ellipse correction tool
- make sure the basket is visible and not heavily obstructed

If batch mode does not process anything:

- confirm your videos are inside `Video Data/`
- confirm the files use a supported video format

## Supported Video Formats

The app currently looks for common video types such as:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.m4v`
- `.wmv`

## About The Project

EspExAnalyser was built as an espresso extraction analysis tool for studying shot quality through video. It combines computer vision, visual review, and data export in one desktop workflow.

## Author

Jephtha Ashter Tandri (20600677)
