import cv2
import numpy as np
import os
from Portafilter_Detection import detect_elliptical_portafilter_with_holes, load_image_with_orientation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Image Data", "Frames")       # Original ROI frames
CROP_DIR = os.path.join(BASE_DIR, "Image Data", "Cropped")      # cropped frames
OUTPUT_DIR = os.path.join(BASE_DIR, "Image Data", "Stabilised")  # stabilised output
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROI_SCALE = 1.5  # how far below the portafilter ellipse to crop (1–2 diameters)



def get_portafilter_reference():
    first_frame_path = os.path.join(INPUT_DIR, "frame_0001.jpg")
    second_frame_path = os.path.join(INPUT_DIR, "frame_0020.jpg")

    if not os.path.exists(first_frame_path):
        raise FileNotFoundError("frame_0001.jpg not found")

    frame1 = load_image_with_orientation(first_frame_path)
    frame20 = (
        load_image_with_orientation(second_frame_path)
        if os.path.exists(second_frame_path)
        else None
    )

    print("[INFO] Detecting portafilter ellipse...")
    _, ellipse = detect_elliptical_portafilter_with_holes(
        frame1,
        save_dashboard=False,
        use_interactive=False,
        second_frame=frame20,
        mask_threshold=15
    )

    if ellipse is None:
        raise RuntimeError("Portafilter detection failed")

    return ellipse


def crop_roi(frame, ellipse, padding=10):
    (cx, cy), (major, minor), _ = ellipse
    h, w = frame.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty ellipse mask")

    x1 = max(xs.min() - padding, 0)
    x2 = min(xs.max() + padding, w)
    y1 = max(ys.min() - padding, 0)
    y2 = h  # keep stream below basket

    return frame[y1:y2, x1:x2]

def load_and_crop_frames(ellipse):
    frame_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".png"))
    )

    cropped_frames = []

    for i, fname in enumerate(frame_files):
        frame = load_image_with_orientation(os.path.join(INPUT_DIR, fname))
        roi = crop_roi(frame, ellipse)
        crop_path = os.path.join(CROP_DIR, f"cropped_{i:04d}.jpg")
        cv2.imwrite(crop_path, roi)
        cropped_frames.append((fname, roi))

    if len(cropped_frames) < 2:
        raise RuntimeError("Not enough frames for cropping")

    return cropped_frames

def stabilise_frames(cropped_frames):
    ref_name, ref_frame = cropped_frames[0]
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    h, w = ref_gray.shape

    # FAST features (good for rim + holes)
    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    kp = fast.detect(ref_gray, None)

    if len(kp) < 20:
        raise RuntimeError("Insufficient FAST keypoints for stabilisation")

    pts_prev = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)
    prev_gray = ref_gray

    stabilised = [(ref_name, ref_frame)]

    for fname, frame in cropped_frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            gray,
            pts_prev,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if pts_curr is None:
            stabilised.append((fname, frame))
            continue

        good_prev = pts_prev[status.flatten() == 1]
        good_curr = pts_curr[status.flatten() == 1]

        if len(good_prev) < 6:
            stabilised.append((fname, frame))
            continue

        # Estimate translation + small rotation
        M, _ = cv2.estimateAffinePartial2D(
            good_curr,
            good_prev,
            method=cv2.RANSAC,
            ransacReprojThreshold=3
        )

        if M is None:
            stabilised.append((fname, frame))
            continue

        aligned = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        stabilised.append((fname, aligned))
        prev_gray = gray
        pts_prev = good_curr.reshape(-1, 1, 2)

    return stabilised


# ============================
# MAIN
# ============================

if __name__ == "__main__":

    print("[INFO] Starting portafilter stabilisation pipeline")

    ellipse = get_portafilter_reference()
    cropped_frames = load_and_crop_frames(ellipse)
    stabilised_frames = stabilise_frames(cropped_frames)

    print("[INFO] Saving stabilised frames...")
    for i, (fname, frame) in enumerate(stabilised_frames):
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"stabilised_{i:04d}.jpg"), frame)

    print(f"[DONE] Stabilised frames saved to {OUTPUT_DIR}")

