import cv2
import os
import numpy as np

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Video_Path_Default = os.path.join(Base_Dir, "Video Data/test1.mp4")
Frame_Dir = os.path.join(Base_Dir, "Image Data/Frames")
Frame_Rate = 1

os.makedirs(Frame_Dir, exist_ok=True)

def preprocess_frame(frame):
    # 1. Auto-rotation correction
    h, w = frame.shape[:2]
    if w > h:  # Landscape → rotate to portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

def extract_frames(video_path, output_dir, target_fps=1):
    # Extract frames at a fixed frame rate.
    os.makedirs(output_dir, exist_ok=True)
    # Keep extraction stable for bad inputs like 0 or negative fps values.
    target_fps = max(1, float(target_fps))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        # Fallback for codecs that do not report source FPS.
        original_fps = target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(original_fps / target_fps))
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
    video_path = sys.argv[1] if len(sys.argv) > 1 else Video_Path_Default
    output_dir = sys.argv[2] if len(sys.argv) > 2 else Frame_Dir
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else Frame_Rate
    
    print(f"Extracting frames from: {video_path}")
    extract_frames(video_path, output_dir, target_fps=fps)
