import cv2
import os
#from ultralytics import YOLO
import numpy as np

# === CONFIGURATION ===
base_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(base_dir, "Video Data/test3.mp4")
FRAME_DIR = os.path.join(base_dir, "Image Data/Frames")
MODEL_PATH = "models/portafilter_yolov8.pt"
FRAME_RATE = 1  # fps for analysis

# === SETUP ===
os.makedirs(FRAME_DIR, exist_ok=True)
# model = YOLO(MODEL_PATH)  # Portafilter detection

# === FRAME EXTRACTION FUNCTION ===
def extract_frames(video_path, output_dir, target_fps=FRAME_RATE):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(original_fps / target_fps)

    print(f"[INFO] Original FPS: {original_fps:.2f}, Target FPS: {target_fps}")
    print(f"[INFO] Total frames in video: {total_frames}")
    print(f"[INFO] Saving every {frame_interval} frames...")

    frame_idx = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"[DONE] Extracted {saved_count} frames to '{output_dir}'")

def detect_portafilter(frame):
    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf > 0.6:  # threshold
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            return frame[y1:y2, x1:x2], (x1, y1, x2, y2)  # cropped ROI + coords
    return None, None

def analyze_frame(frame, portafilter_bbox):
    if portafilter_bbox is None:
        return

    x1, y1, x2, y2 = portafilter_bbox
    stream_roi = frame[y2:y2+100, x1:x2]  # Below the portafilter
    lab = cv2.cvtColor(stream_roi, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab)[:3]
    return mean_lab, stream_roi

def run_pipeline():
    extract_frames(VIDEO_PATH, FRAME_DIR)
    for filename in sorted(os.listdir(FRAME_DIR)):
        path = os.path.join(FRAME_DIR, filename)
        frame = cv2.imread(path)

        # portafilter_roi, bbox = detect_portafilter(frame)
        # if bbox is not None:
        #     mean_lab, stream_roi = analyze_frame(frame, bbox)

        #     print(f"{filename} → L={mean_lab[0]:.2f}, A={mean_lab[1]:.2f}, B={mean_lab[2]:.2f}")
        #     # TODO: Save annotated frame, compute optical flow, track over time

if __name__ == "__main__":
    run_pipeline()
