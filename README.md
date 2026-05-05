# Espresso Shot Extraction Analysis Tool

EspExAnalyser is an espresso extraction analysis system for recorded bottomless-portafilter videos. It can detect the portafilter basket, track the extraction area through the shot, measure blonding and channeling behaviour, generate diagnostics, export datasets, and train or run an extraction-quality classifier.

The project is intended to work as both:

- a desktop analysis tool for manually reviewing espresso shots
- a backend analysis engine for the CeKopi mobile app or other client applications that upload videos through the API

## Repository And Dataset

Project repository:

```text
https://github.com/Jephtha-T/EspExAnalyser
```

The labelled dataset logs will be available from `dataset.csv`, while all videos in the dataset are available in the following Google Drive link:

```text
https://drive.google.com/drive/folders/18V3Gf2je1byH3I541jVSaMWWNYtMkDEE?usp=drive_link
```

## Main Capabilities

- Analyse one espresso video at a time
- Batch analyse every video in `Video Data/`
- Automatically detect the portafilter basket and allow for manual correction of the basket ROI when automatic detection is unsatisfactory
- Track the approved ROI through the shot
- Extract cropped analysis frames
- Measure blonding progression
- Measure channeling and visible stream irregularities
- Compute diagnostic flags from rule-based metrics
- Predict extraction class with trained ML models
- Export summary, event, and frame-level CSV datasets
- Expose a FastAPI backend for mobile or web clients
- Include the CeKopi Flutter app source and platform runner projects

## Requirements

- Python 3.10 or newer
- Git
- A folder of espresso extraction videos
- A working Python environment with the required packages
- Flutter SDK for running the CeKopi mobile app

Install the Python dependencies from inside the `EspExAnalyser` folder:

```bash
pip install fastapi joblib matplotlib numpy opencv-python openpyxl pandas pillow pydantic scikit-learn ultralytics uvicorn
```

Notes:

- `tkinter` is required for the desktop interface.
- `ultralytics` enables YOLO-based portafilter detection.
- `portafilter_latest.pt` is the YOLO model file used by the detector when available.

## Project Layout

```text
EspExAnalyser/
  Desktop_Runner.py              desktop user interface
  Api_Server.py                  FastAPI server for uploaded-video workflows
  Espresso_Analysis.py           shared pipeline orchestration
  Frame_Extraction.py            video frame sampling
  Portafilter_Detection.py       basket detection and manual ROI helpers
  Portafilter_Tracking_v2.py     ROI tracking and crop generation
  Feature_Extraction.py          blonding, channeling, quality, and plots
  Espresso_Diagnostics.py        rule-based diagnostic flags
  espresso_diagnostic_rules.json diagnostic rule definitions
  Data_Export.py                 CSV dataset export
  Espresso_Model.py              model training and prediction
  dataset.csv                    labelled video metadata
  cekopi/                        Flutter app source and platform runners
  Video Data/                    local input videos, ignored by Git
  Image Data/                    generated frames and crops, ignored by Git
  Analysis/                      generated results, plots, models, and CSVs
  api_storage/                   API upload/session storage, ignored by Git
```

Generated runtime folders are created automatically when the system runs.

## Label Format

`dataset.csv` links video files to extraction labels. The expected columns include:

```text
Id,Video Name,Completion time,Espresso Extraction Level,Was There Channeling, Spraying or Fast Blonding?
```

Supported extraction labels are:

- `Ideal`
- `Under-Extracted`
- `Over-Extracted`

During export, labels are converted to numeric classes:

- `0` = under-extracted
- `1` = ideal
- `2` = over-extracted

Make sure the `Video Name` values match the video filenames or result JSON names.

## Desktop Manual

Start the desktop app from the project folder:

```bash
python Desktop_Runner.py
```

### Single Video Analysis

1. Put the video anywhere on your computer, or place it in `Video Data/`.
2. Open the desktop app.
3. Select `Single`.
4. Click `Browse Video` and choose the video.
5. Click `Start Analysis`.
6. Review the detected portafilter ROI.
7. If the ROI is correct, continue with the automatic detection.
8. If the ROI is wrong, choose the manual ROI option and draw an ellipse around the basket.
9. Wait for frame extraction, ROI tracking, feature extraction, diagnostics, and result rendering.
10. Review the summary, charts, replay frames, quality score, diagnostic flags, and ML prediction if a model is available.
11. Use `Export Figures / Replay` to save visual reports.
12. Use `Export Results CSV` to refresh the CSV dataset exports.

### Batch Analysis

1. Put all videos to analyse in:

```text
Video Data/
```

2. Open the desktop app:

```bash
python Desktop_Runner.py
```

3. Select `Batch (all videos in Video Data)`.
4. Click `Start Analysis`.
5. Wait for each video to be processed.
6. Review the batch summary.
7. Click `Export Results CSV` when batch processing is finished.

Batch mode uses automatic ROI detection. Use single mode first if you need to check whether your videos are framed well enough for automatic detection.

## CeKopi Flutter App

The `cekopi/` folder contains the source code and platform runner projects for the CeKopi Flutter mobile app. It is included in this repository so the mobile client can be opened, run, and modified together with the EspExAnalyser backend.

CeKopi is responsible for:

- showing the splash, home, history, and coffee-library screens
- recording espresso extraction videos with the device camera
- uploading recorded videos to the EspExAnalyser API
- showing the detected ROI on the uploaded video
- letting the user accept or redraw the ROI ellipse
- starting backend analysis
- polling backend progress
- displaying the final analysis results
- saving analysis sessions locally with bean details

### CeKopi Folder Layout

```text
cekopi/
  lib/
    main.dart
    app.dart
    core/
      assets/                    asset constants
      config/backend_config.dart backend URL and debug flags
      navigation/                route names and router
      theme/                     app colours
      utils/                     JSON helpers
    features/
      splash/                    splash screen
      home/                      home/menu screen
      video_capture/             camera recording and upload service
      analysis/                  ROI, processing, result, history, library
      shared/                    shared widgets
  assets/                        app logo and cafe illustrations
  android/                       Android runner project
  ios/                           iOS runner project
  web/                           Web runner project
  windows/ linux/ macos/         desktop runner projects
  pubspec.yaml                   Flutter package/dependency config
  pubspec.lock                   locked dependency versions
```

Generated Flutter folders such as `cekopi/.dart_tool/` and `cekopi/build/` are ignored and should not be committed.

### CeKopi Requirements

Install:

- Flutter SDK
- Android Studio or Android command-line tools
- An Android emulator or physical Android device
- Xcode if running on iOS or macOS

Check the Flutter setup from inside the app folder:

```bash
cd cekopi
flutter doctor
flutter pub get
```

The app currently uses Dart SDK `^3.11.4` as declared in `cekopi/pubspec.yaml`.

### Running CeKopi With The Backend

Start the Python backend first from the main project folder:

```bash
python Api_Server.py
```

Then run the Flutter app from the `cekopi/` folder:

```bash
cd cekopi
flutter run
```

The default API URL is configured for the Android emulator:

```text
http://10.0.2.2:8000
```

For iOS simulator, desktop, or web, use localhost:

```bash
flutter run --dart-define=ESPRESSO_API_BASE_URL=http://127.0.0.1:8000
```

For a physical phone, use the computer's LAN IP address. The phone and computer must be on the same network:

```bash
flutter run --dart-define=ESPRESSO_API_BASE_URL=http://192.168.1.10:8000
```

Replace `192.168.1.10` with the IP address of the computer running `Api_Server.py`.

Backend settings are defined in:

```text
cekopi/lib/core/config/backend_config.dart
```

Important compile-time flags:

- `ESPRESSO_API_BASE_URL` sets the API base URL.
- `USE_MOCK_ANALYSIS_FLOW` enables a no-backend demo flow.
- `CEKOPI_TEST_VIDEO_PATH` sets the emulator test-video path.

### Mock Analysis Mode

Run CeKopi without the Python backend:

```bash
flutter run --dart-define=USE_MOCK_ANALYSIS_FLOW=true
```

Mock mode returns placeholder ROI and analysis data. Use it for UI testing, route testing, and demonstration checks only. It does not analyse the uploaded video.

### Emulator Test Video

The video capture page includes an emulator test-video upload path for debugging. The default path is:

```text
/storage/emulated/0/Android/data/com.example.cekopi/files/cekopi_test.mov
```

Push a test video to the Android emulator:

```bash
adb push path/to/test-video.mov /storage/emulated/0/Android/data/com.example.cekopi/files/cekopi_test.mov
```

Or override the path at runtime:

```bash
flutter run --dart-define=CEKOPI_TEST_VIDEO_PATH=/storage/emulated/0/Android/data/com.example.cekopi/files/my_video.mov
```

### CeKopi User Flow

1. Open CeKopi.
2. Choose the video capture flow from the home screen.
3. Record a clear espresso extraction video.
4. CeKopi uploads the video to the backend.
5. Review the detected ROI on the video preview.
6. Accept the ROI or drag a new ellipse around the portafilter basket.
7. CeKopi starts analysis and shows backend progress.
8. Review the result page, including metrics, curves, flags, and video playback.
9. Save the result with bean name, bean type, and roast level.
10. Revisit saved entries through History or Coffee Library.

Saved entries are stored locally with `shared_preferences`, so they are device-local app data rather than files in the Git repository.

The Android runner includes camera, microphone, and internet permissions. It also enables cleartext HTTP traffic so the app can call a local development backend over `http://`.

## CeKopi API Workflow

`Api_Server.py` exposes the analysis pipeline for CeKopi or any other client that needs to upload a video and poll for results.

Start the API server:

```bash
python Api_Server.py
```

The server runs on:

```text
http://localhost:8000
```

Health check:

```text
GET /health
```

### API Request Flow

1. In CeKopi, choose or record an espresso extraction video.
2. The client uploads the video to:

```text
POST /api/analyse-video/upload
```

3. The API stores the upload in `api_storage/`, extracts preview frames, and returns a `session_id`.
4. The client shows the detected ROI to the user.
5. The user accepts the automatic ROI or adjusts it.
6. The client starts full analysis with:

```text
POST /api/analyse-video/start
```

7. The client polls progress with:

```text
GET /api/analyse-video/status?job_id=<session_id>
```

8. The client can request generated preview/replay frames with:

```text
GET /api/analyse-video/frame?job_id=<session_id>&name=<frame_name>
```

9. When the status becomes `completed`, the response contains the analysis summary, quality score, diagnostics, prediction data when available, and frame references.

The API accepts ROI ellipses as either normalized values from `0` to `1` or pixel values. The required ellipse fields are `cx`, `cy`, `width`, `height`, and optionally `angle_deg`.

## Analysis Pipeline

Every desktop and API run follows the same core pipeline:

1. Build or create a workspace.
2. Extract preview frames from the video.
3. Detect the portafilter ROI.
4. Ask the user or client to approve/correct the ROI.
5. Extract working frames at the configured sampling FPS.
6. Track the approved ROI through the shot.
7. Crop/stabilise the basket region.
8. Extract blonding and channeling features.
9. Save plots, JSON results, and optional replay images.
10. Compute diagnostic flags and quality scores.
11. Run the trained ML model if a model file is available.
12. Export datasets when requested.

## Output Files

Per-video JSON results are saved to:

```text
Analysis/results/<video_name>_results.json
```

Feature plots and channeling/blonding assets are saved under:

```text
Analysis/blond/
Analysis/channeling/
```

Exported CSV files are saved to:

```text
Analysis/training_data.csv
Analysis/training_data_events.csv
Analysis/training_data_timeseries.csv
```

The CSV files have different purposes:

- `training_data.csv` contains one summary row per video.
- `training_data_events.csv` contains event/timing features such as blonding and channeling threshold times.
- `training_data_timeseries.csv` contains frame-level feature rows.

## Exporting Dataset CSVs

From the desktop app, click `Export Results CSV`.

From the command line:

```bash
python Data_Export.py
```

The exporter reads:

```text
Analysis/results/
dataset.csv
```

and writes the training CSV files into `Analysis/`.

If a result file has no matching label in `dataset.csv`, the exporter will warn that the video has no label match.

## Training Models

Train the extraction classifier from exported CSV files:

```bash
python Espresso_Model.py train
```

The default training command reads:

```text
Analysis/training_data.csv
Analysis/training_data_events.csv
```

By default, the script compares supported model families and writes model files into `Analysis/`, including:

```text
Analysis/extraction_model.joblib
Analysis/extraction_model_rf.joblib
Analysis/extraction_model_svm.joblib
Analysis/extraction_model_logistic.joblib
```

Useful training options:

```bash
python Espresso_Model.py train --model-type all --feature-set auto
python Espresso_Model.py train --model-type svm --feature-set summary
python Espresso_Model.py train --test-size 0.2 --random-state 42
```

Supported model types:

- `all`
- `both`
- `svm`
- `random_forest`
- `logistic`

## Predicting From Results

After analysing a video and training a model, predict from a result JSON file:

```bash
python Espresso_Model.py predict --results-json Analysis/results/<video_name>_results.json
```

To print the full prediction payload:

```bash
python Espresso_Model.py predict --results-json Analysis/results/<video_name>_results.json --print-json
```

## Recommended End-To-End Workflow

1. Store raw videos locally in `Video Data/`.
2. Keep `Video Data/` out of Git.
3. Add or update labels in `dataset.csv`.
4. Run single-video analysis first to check framing and ROI detection.
5. Use manual ROI correction when needed.
6. Run batch analysis after the setup is reliable.
7. Export CSV datasets.
8. Train or retrain the classifier.
9. Run predictions on new analysis results.
10. Commit code, rules, docs, dataset labels, the `cekopi/` source folder, and small generated CSV/model outputs as needed.

## Input Video Setup

Best results usually come from videos that:

- clearly show the bottom of the portafilter
- keep the basket in frame throughout the shot
- use stable camera framing
- avoid strong reflections and motion blur
- keep the stream and cup visible
- use consistent lighting

If the detected ROI is inaccurate, use manual ROI correction in single mode before trusting a batch run.

## Supported Video Formats

The desktop app looks for common video types including:

- `.mp4`
- `.mov`
- `.avi`
- `.mkv`
- `.m4v`
- `.wmv`

## Troubleshooting

### The Desktop App Does Not Open

- Confirm Python 3.10+ is installed.
- Install the dependency packages.
- Check that `tkinter` is available.
- Run the app from inside the `EspExAnalyser` folder.

### Portafilter Detection Fails

- Use a clearer first frame.
- Improve lighting.
- Make sure the basket is not blocked.
- Use manual ROI correction.
- Confirm `portafilter_latest.pt` is present if YOLO detection is expected.

### Batch Mode Does Nothing

- Confirm videos are inside `Video Data/`.
- Confirm video extensions are supported.
- Try one file in single mode first.

### CSV Export Has Missing Labels

- Confirm `dataset.csv` exists in the project folder.
- Confirm the `Video Name` column matches analysed video filenames.
- Re-run `Export Results CSV` after fixing labels.

### CeKopi Cannot Reach The Backend

- Confirm `python Api_Server.py` is running.
- Open `http://localhost:8000/health` on the backend computer.
- Use `http://10.0.2.2:8000` for the Android emulator.
- Use `http://127.0.0.1:8000` for iOS simulator, desktop, or web.
- Use the backend computer's LAN IP address for a physical phone.
- Confirm the phone and backend computer are on the same network.

## Author

Jephtha Ashter Tandri (20600677)
