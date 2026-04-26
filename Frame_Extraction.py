import cv2
import math
import os
import shutil
import subprocess
import numpy as np

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Video_Path_Default = os.path.join(Base_Dir, "Video Data/test1.mp4")
Frame_Dir = os.path.join(Base_Dir, "Image Data/Frames")
DEFAULT_EXTRACTION_FPS = 1.0
PREVIEW_FRAME_OFFSET_SECONDS = 10.0
Frame_Rate = DEFAULT_EXTRACTION_FPS

os.makedirs(Frame_Dir, exist_ok=True)


def safe_fps(fps, default=DEFAULT_EXTRACTION_FPS):
    try:
        value = float(fps)
    except (TypeError, ValueError):
        value = float(default)

    if not math.isfinite(value) or value <= 0.0:
        value = float(default)
    return float(value)


def sampled_frame_offset_for_seconds(seconds, fps=DEFAULT_EXTRACTION_FPS):
    try:
        seconds_value = float(seconds)
    except (TypeError, ValueError):
        seconds_value = 0.0
    seconds_value = max(0.0, seconds_value)
    return int(round(seconds_value * safe_fps(fps)))


def source_frame_index_for_seconds(seconds, source_fps, total_frames):
    total_frames = max(0, int(total_frames or 0))
    if total_frames == 0:
        return 0

    frame_offset = sampled_frame_offset_for_seconds(seconds, fps=safe_fps(source_fps, default=1.0))
    return min(frame_offset, total_frames - 1)


def analysed_frame_count(flow_start, flow_end):
    try:
        start_frame = int(flow_start)
    except (TypeError, ValueError):
        start_frame = 0
    try:
        end_frame = int(flow_end)
    except (TypeError, ValueError):
        end_frame = start_frame
    return max(0, end_frame - start_frame)


def shot_duration_seconds(flow_start, flow_end, fps=DEFAULT_EXTRACTION_FPS):
    return float(analysed_frame_count(flow_start, flow_end)) / safe_fps(fps)


def estimated_sample_count(total_frames, source_fps, target_fps=DEFAULT_EXTRACTION_FPS):
    total_frames = max(0, int(total_frames or 0))
    if total_frames == 0:
        return 0

    source_fps = safe_fps(source_fps, default=1.0)
    target_fps = safe_fps(target_fps)
    last_frame_time_s = max(0.0, float(total_frames - 1) / source_fps)
    return int(math.floor(last_frame_time_s * target_fps + 1e-9)) + 1

def preprocess_frame(frame):
    # 1. Auto-rotation correction
    h, w = frame.shape[:2]
    if w > h:  # Landscape → rotate to portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame


def _get_video_sampling_info(video_path, target_fps=DEFAULT_EXTRACTION_FPS):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    target_fps = safe_fps(target_fps)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = float(target_fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(float(original_fps) / target_fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "original_fps": float(original_fps),
        "total_frames": int(total_frames),
        "frame_interval": int(frame_interval),
        "duration_seconds": float(total_frames / float(original_fps)) if total_frames > 0 else 0.0,
        "sample_count": int(estimated_sample_count(total_frames, original_fps, target_fps)),
        "width": width,
        "height": height,
    }


def read_video_frames_by_index(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    output = {}
    for frame_index in sorted({max(0, int(index)) for index in frame_indices}):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        output[frame_index] = preprocess_frame(frame)

    cap.release()
    return output


def load_preview_frames(
    video_path,
    target_fps=DEFAULT_EXTRACTION_FPS,
    second_offset_seconds=PREVIEW_FRAME_OFFSET_SECONDS,
):
    info = _get_video_sampling_info(video_path, target_fps=target_fps)
    if info["total_frames"] <= 0:
        raise RuntimeError("Video has no readable frames.")

    second_frame_idx = source_frame_index_for_seconds(
        second_offset_seconds,
        info["original_fps"],
        info["total_frames"],
    )
    requested_indices = [0, second_frame_idx]
    frames = read_video_frames_by_index(video_path, requested_indices)
    first_frame = frames.get(0)
    second_frame = frames.get(second_frame_idx, first_frame)
    if first_frame is None:
        raise RuntimeError("Failed to load preview frame from video.")
    return {
        "first_frame": first_frame,
        "second_frame": second_frame,
        "sampling": info,
        "second_frame_index": second_frame_idx,
    }


def _extract_frames_opencv(video_path, output_dir, target_fps=DEFAULT_EXTRACTION_FPS):
    info = _get_video_sampling_info(video_path, target_fps=target_fps)
    original_fps = info["original_fps"]
    total_frames = info["total_frames"]
    target_fps = safe_fps(target_fps)
    sample_interval_s = 1.0 / target_fps

    cap = cv2.VideoCapture(video_path)
    print(f"Original FPS: {original_fps:.2f} | Target FPS: {target_fps:.2f}")
    print(f"Total frames in video: {total_frames}")
    print(f"Extracting frames at {target_fps:.2f} fps...")

    frame_idx = 0
    saved_count = 0
    next_sample_time_s = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_time_s = float(frame_idx) / max(1e-6, float(original_fps))
        if frame_time_s + 1e-9 >= next_sample_time_s:
            # Preprocess and save
            processed = preprocess_frame(frame)
            out_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, processed)
            saved_count += 1
            next_sample_time_s += sample_interval_s

        frame_idx += 1

    cap.release()
    print(f"Extracted and saved {saved_count} frames to '{output_dir}'")
    return saved_count


def _extract_frames_ffmpeg(video_path, output_dir, target_fps=DEFAULT_EXTRACTION_FPS):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return None

    info = _get_video_sampling_info(video_path, target_fps=target_fps)
    width = int(info["width"] or 0)
    height = int(info["height"] or 0)

    filters = [f"fps={safe_fps(target_fps):.6g}"]
    if width > height and height > 0:
        filters.append("transpose=2")
    vf = ",".join(filters)
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-vf",
        vf,
        "-start_number",
        "0",
        "-q:v",
        "2",
        output_pattern,
    ]

    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        print(f"FFmpeg extraction failed, falling back to OpenCV: {detail}")
        return None

    saved_count = len(
        [
            name
            for name in os.listdir(output_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )
    print(f"Extracted and saved {saved_count} frames to '{output_dir}' with FFmpeg")
    return saved_count


def extract_frames(video_path, output_dir, target_fps=DEFAULT_EXTRACTION_FPS):
    # Extract frames at a fixed frame rate.
    os.makedirs(output_dir, exist_ok=True)
    target_fps = safe_fps(target_fps)

    ffmpeg_result = _extract_frames_ffmpeg(video_path, output_dir, target_fps=target_fps)
    if ffmpeg_result is not None:
        return ffmpeg_result
    return _extract_frames_opencv(video_path, output_dir, target_fps=target_fps)

if __name__ == "__main__":
    import sys
    
    # Allow command-line usage  
    video_path = sys.argv[1] if len(sys.argv) > 1 else Video_Path_Default
    output_dir = sys.argv[2] if len(sys.argv) > 2 else Frame_Dir
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else Frame_Rate
    
    print(f"Extracting frames from: {video_path}")
    extract_frames(video_path, output_dir, target_fps=fps)
