import cv2
import os
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path_default = os.path.join(base_dir, "Video Data/test1.mp4")
frame_dir = os.path.join(base_dir, "Image Data/Frames")
frame_rate = 1

os.makedirs(frame_dir, exist_ok=True)

def preprocess_frame(frame):
    # 1. Auto-rotation correction
    h, w = frame.shape[:2]
    if w > h:  # Landscape → rotate to portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

def extract_frames(video_path, output_dir, target_fps=1):
    # Extract frames at a fixed frame rate.
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(original_fps / target_fps)
    print(f"Original FPS: {original_fps:.2f} | Target FPS: {target_fps}")
    print(f"Total frames in video: {total_frames}")
    print(f"Extracting every {frame_interval} frames...")

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Preprocess and save
            processed = preprocess_frame(frame)
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, processed)
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"Extracted and saved {saved_count} frames to '{output_dir}'")
    return saved_count

if __name__ == "__main__":
    import sys
    
    # Allow command-line usage  
    video_path = sys.argv[1] if len(sys.argv) > 1 else video_path_default
    output_dir = sys.argv[2] if len(sys.argv) > 2 else frame_dir
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else frame_rate
    
    print(f"Extracting frames from: {video_path}")
    extract_frames(video_path, output_dir, target_fps=fps)
