import cv2
import os
#from ultralytics import YOLO
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

# === CONFIGURATION ===
base_dir = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(base_dir, "Video Data/test5.mp4")
FRAME_DIR = os.path.join(base_dir, "Image Data/Frames")
MODEL_PATH = "models/portafilter_yolov8.pt"
FRAME_RATE = 1  # fps for analysis

# === SETUP ===
os.makedirs(FRAME_DIR, exist_ok=True)
# model = YOLO(MODEL_PATH)  # Portafilter detection

def calculate_frame_quality(frame):
    """
    Calculate frame quality score based on sharpness, brightness, and contrast.
    
    Args:
        frame: Input frame (BGR)
    
    Returns:
        quality_score: Higher is better
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Brightness (mean intensity)
    brightness = np.mean(gray)
    
    # 3. Contrast (standard deviation)
    contrast = np.std(gray)
    
    # 4. Focus measure (Tenengrad)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(sobelx**2 + sobely**2)
    
    # Combine metrics (normalize and weight)
    quality_score = (
        0.4 * min(float(laplacian_var) / 500, 1.0) +  # Sharpness
        0.2 * min(float(brightness) / 128, 1.0) +      # Brightness
        0.2 * min(float(contrast) / 50, 1.0) +         # Contrast
        0.2 * min(float(tenengrad) / 1000, 1.0)        # Focus
    )
    
    return quality_score

def detect_shot_boundaries(frames, threshold=0.3):
    """
    Detect shot boundaries using frame difference analysis.
    
    Args:
        frames: List of frames
        threshold: Difference threshold for shot detection
    
    Returns:
        shot_boundaries: List of frame indices where shots change
    """
    shot_boundaries = [0]  # First frame is always a boundary
    
    for i in range(1, len(frames)):
        # Calculate frame difference
        diff = cv2.absdiff(frames[i], frames[i-1])
        mean_diff = np.mean(diff)
        
        # Normalize by frame size
        frame_size = frames[i].shape[0] * frames[i].shape[1]
        normalized_diff = mean_diff / frame_size
        
        if normalized_diff > threshold:
            shot_boundaries.append(i)
    
    return shot_boundaries

def select_representative_frames(frames, shot_boundaries, max_frames_per_shot=3):
    """
    Select representative frames from each shot based on quality.
    
    Args:
        frames: List of all frames
        shot_boundaries: List of shot boundary indices
        max_frames_per_shot: Maximum frames to select per shot
    
    Returns:
        selected_frames: List of (frame_index, frame) tuples
    """
    selected_frames = []
    
    for i in range(len(shot_boundaries)):
        start_idx = shot_boundaries[i]
        end_idx = shot_boundaries[i + 1] if i + 1 < len(shot_boundaries) else len(frames)
        
        # Get frames in this shot
        shot_frames = frames[start_idx:end_idx]
        
        # Calculate quality scores for all frames in shot
        quality_scores = []
        for j, frame in enumerate(shot_frames):
            quality = calculate_frame_quality(frame)
            quality_scores.append((start_idx + j, quality))
        
        # Sort by quality and select top frames
        quality_scores.sort(key=lambda x: x[1], reverse=True)
        num_to_select = min(max_frames_per_shot, len(quality_scores))
        
        for k in range(num_to_select):
            frame_idx, quality = quality_scores[k]
            selected_frames.append((frame_idx, frames[frame_idx], quality))
    
    return selected_frames

def preprocess_frame(frame):
    """
    Apply preprocessing to improve frame quality for analysis.
    
    Args:
        frame: Input frame
    
    Returns:
        processed_frame: Preprocessed frame
    """
    # 1. Auto-rotation detection and correction
    h, w = frame.shape[:2]
    if w > h:  # Landscape orientation
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # 2. Auto-brightness and contrast adjustment
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    lab = cv2.merge([l, a, b])
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 3. Noise reduction
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
    
    # 4. Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv2.filter2D(frame, -1, kernel)
    
    return frame

def extract_frames_improved(video_path, output_dir, target_fps=FRAME_RATE, 
                          enable_shot_detection=True, enable_quality_selection=True):
    """
    Improved frame extraction with shot detection and quality-based selection.
    
    Args:
        video_path: Path to input video
        output_dir: Output directory for frames
        target_fps: Target frame rate
        enable_shot_detection: Whether to detect shot boundaries
        enable_quality_selection: Whether to select frames based on quality
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(original_fps / target_fps)

    print(f"[INFO] Original FPS: {original_fps:.2f}, Target FPS: {target_fps}")
    print(f"[INFO] Total frames in video: {total_frames}")
    print(f"[INFO] Frame interval: {frame_interval}")

    # Extract all frames first
    all_frames = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Preprocess frame
            processed_frame = preprocess_frame(frame)
            all_frames.append(processed_frame)
        
        frame_idx += 1
    
    cap.release()
    print(f"[INFO] Extracted {len(all_frames)} frames for analysis")

    # Shot detection
    if enable_shot_detection and len(all_frames) > 1:
        print("[INFO] Detecting shot boundaries...")
        shot_boundaries = detect_shot_boundaries(all_frames)
        print(f"[INFO] Found {len(shot_boundaries)} shots")
    else:
        shot_boundaries = [0, len(all_frames)]

    # Quality-based frame selection
    if enable_quality_selection:
        print("[INFO] Selecting representative frames based on quality...")
        selected_frames = select_representative_frames(all_frames, shot_boundaries)
    else:
        # Use all frames if quality selection is disabled
        selected_frames = [(i, frame, calculate_frame_quality(frame)) 
                          for i, frame in enumerate(all_frames)]

    # Save selected frames
    saved_count = 0
    for frame_idx, frame, quality in selected_frames:
        out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(out_path, frame)
        
        # Save quality info
        quality_path = os.path.join(output_dir, f"frame_{saved_count:04d}_quality.txt")
        with open(quality_path, 'w') as f:
            f.write(f"Original frame index: {frame_idx}\n")
            f.write(f"Quality score: {quality:.4f}\n")
        
        saved_count += 1

    print(f"[DONE] Saved {saved_count} representative frames to '{output_dir}'")
    print(f"[INFO] Average quality score: {np.mean([q for _, _, q in selected_frames]):.4f}")

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

        # Check if rotation is needed (portrait frame with landscape shape)
        h, w = frame.shape[:2]
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # or ROTATE_90_COUNTERCLOCKWISE if needed

        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"[DONE] Extracted {saved_count} frames to '{output_dir}'")

# def detect_portafilter(frame):
#     results = model(frame)[0]
#     for box in results.boxes:
#         cls = int(box.cls[0])
#         conf = float(box.conf[0])
#         if conf > 0.6:  # threshold
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             return frame[y1:y2, x1:x2], (x1, y1, x2, y2)  # cropped ROI + coords
#     return None, None

def analyze_frame(frame, portafilter_bbox):
    if portafilter_bbox is None:
        return

    x1, y1, x2, y2 = portafilter_bbox
    stream_roi = frame[y2:y2+100, x1:x2]  # Below the portafilter
    lab = cv2.cvtColor(stream_roi, cv2.COLOR_BGR2LAB)
    mean_lab = cv2.mean(lab)[:3]
    return mean_lab, stream_roi

def run_pipeline():
    # Use improved frame extraction
    extract_frames_improved(VIDEO_PATH, FRAME_DIR, enable_shot_detection=True, enable_quality_selection=True)
    
    for filename in sorted(os.listdir(FRAME_DIR)):
        if filename.endswith('.jpg'):
            path = os.path.join(FRAME_DIR, filename)
            frame = cv2.imread(path)

            # portafilter_roi, bbox = detect_portafilter(frame)
            # if bbox is not None:
            #     mean_lab, stream_roi = analyze_frame(frame, bbox)

            #     print(f"{filename} → L={mean_lab[0]:.2f}, A={mean_lab[1]:.2f}, B={mean_lab[2]:.2f}")
            #     # TODO: Save annotated frame, compute optical flow, track over time

if __name__ == "__main__":
    run_pipeline()
