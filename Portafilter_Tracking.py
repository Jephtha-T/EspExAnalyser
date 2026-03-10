import cv2
import numpy as np
import os
from Portafilter_Detection import detect_elliptical_portafilter_with_holes, load_image_with_orientation

base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "Image Data", "Frames")
crop_dir = os.path.join(base_dir, "Image Data", "Cropped")
os.makedirs(crop_dir, exist_ok=True)


def get_portafilter_reference(frames_dir, manual_roi=False, manual_ellipse=None):
    # Detect a reference ellipse from early frames.
    first_frame_path = os.path.join(frames_dir, "frame_0001.jpg")
    second_frame_path = os.path.join(frames_dir, "frame_0020.jpg")

    if not os.path.exists(first_frame_path):
        # Try alternate naming
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png"))])
        if len(frame_files) == 0:
            raise FileNotFoundError("No frames found in directory")
        first_frame_path = os.path.join(frames_dir, frame_files[0])
        second_frame_path = os.path.join(frames_dir, frame_files[min(20, len(frame_files)-1)])

    frame1 = load_image_with_orientation(first_frame_path)
    frame20 = load_image_with_orientation(second_frame_path)

    print("Detecting portafilter ellipse...")
    _, ellipse, hole_mode_size, fast_params = detect_elliptical_portafilter_with_holes(
        frame1,
        save_dashboard=False,
        use_interactive=False,
        second_frame=frame20,
        mask_threshold=15,
        manual_roi=manual_roi,
        manual_ellipse=manual_ellipse,
    )

    if ellipse is None:
        raise RuntimeError("Portafilter detection failed")

    print("Portafilter ellipse detected")
    print(f"Hole mode size: {hole_mode_size}")

    return ellipse, hole_mode_size


def get_crop_bounds(frame_shape, ellipse, padding=12):
    # Build crop bounds around the detected ellipse.
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty ellipse mask")

    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, w)
    y1 = max(int(ys.min()) - padding, 0)
    y2 = h  # keep stream below basket

    return x1, y1, x2, y2


def crop_roi(frame, crop_bounds):
    # Crop frame using precomputed bounds.
    x1, y1, x2, y2 = crop_bounds
    return frame[y1:y2, x1:x2]


def crop_frames(frames_dir, output_dir, ellipse):
    # Load frames and save a fixed ROI crop for each frame.
    frame_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if len(frame_files) < 2:
        raise RuntimeError("Not enough frames for processing")

    print("Loading and cropping frames with fixed ROI bounds...")
    first_frame = load_image_with_orientation(os.path.join(frames_dir, frame_files[0]))
    crop_bounds = get_crop_bounds(first_frame.shape, ellipse, padding=12)

    x1, y1, x2, y2 = crop_bounds
    target_h = y2 - y1
    target_w = x2 - x1

    cropped_frames = []
    for fname in frame_files:
        frame = load_image_with_orientation(os.path.join(frames_dir, fname))
        roi = crop_roi(frame, crop_bounds)
        if roi.shape[:2] != (target_h, target_w):
            roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        cropped_frames.append((fname, roi))

    print(f"Loaded {len(cropped_frames)} cropped frames")

    # Save cropped frames
    print("Saving cropped frames...")
    os.makedirs(output_dir, exist_ok=True)
    for i, (fname, frame) in enumerate(cropped_frames):
        output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(output_path, frame)

    print(f"Saved {len(cropped_frames)} cropped frames to {output_dir}")
    return cropped_frames


def process_portafilter_tracking(frames_dir=None, output_dir=None, manual_roi=False, manual_ellipse=None):
    # Detect the basket and crop all frames to a stable ROI.
    if frames_dir is None:
        frames_dir = input_dir
    if output_dir is None:
        output_dir = crop_dir

    print(f"Input frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")

    # Step 1: Detect portafilter ellipse
    ellipse, hole_mode_size = get_portafilter_reference(
        frames_dir,
        manual_roi=manual_roi,
        manual_ellipse=manual_ellipse,
    )

    # Step 2: Crop frames
    cropped_frames = crop_frames(frames_dir, output_dir, ellipse)

    return {
        "ellipse": ellipse,
        "hole_mode_size": hole_mode_size,
        "output_dir": output_dir,
        "frame_count": len(cropped_frames)
    }


if __name__ == "__main__":
    result = process_portafilter_tracking()
    print("\n===== PORTAFILTER TRACKING COMPLETE =====")
    print(f"Ellipse: {result['ellipse']}")
    print(f"Hole mode size: {result['hole_mode_size']}")
    print(f"Frames processed: {result['frame_count']}")


