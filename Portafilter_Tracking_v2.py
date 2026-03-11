import os
import cv2
import numpy as np
from Portafilter_Detection import detect_elliptical_portafilter_with_holes, load_image_with_orientation
from Frame_Extraction import extract_frames


Base_Dir = os.path.dirname(os.path.abspath(__file__))
Input_Dir = os.path.join(Base_Dir, "Image Data", "Frames")
Crop_Dir = os.path.join(Base_Dir, "Image Data", "Cropped")
os.makedirs(Crop_Dir, exist_ok=True)


def list_frame_files(frames_dir):
    return sorted(
        [name for name in os.listdir(frames_dir) if name.lower().endswith((".jpg", ".png", ".jpeg"))]
    )


def load_all_frames(frames_dir, frame_files):
    frames = []
    for name in frame_files:
        frame_path = os.path.join(frames_dir, name)
        frame = load_image_with_orientation(frame_path)
        if frame is None:
            continue
        frames.append(frame)
    return frames


def estimate_ecc_euclidean(template_gray, input_gray):
    # ECC gives robust global alignment for small camera motion.
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )

    try:
        cv2.findTransformECC(
            template_gray,
            input_gray,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            1,
        )
        return warp_matrix
    except cv2.error:
        return None


def estimate_affine_with_optical_flow(input_gray, template_gray):
    # Fallback when ECC fails: sparse optical flow + robust affine fit.
    points = cv2.goodFeaturesToTrack(
        input_gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points is None or len(points) < 12:
        return None

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        input_gray,
        template_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return None

    valid = status.reshape(-1) == 1
    if int(np.sum(valid)) < 10:
        return None

    src_pts = points[valid].reshape(-1, 2)
    dst_pts = tracked[valid].reshape(-1, 2)
    affine, _ = cv2.estimateAffinePartial2D(
        src_pts,
        dst_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.99,
    )
    return affine


def stabilise_frames(frames):
    # Stabilise all frames to frame 0 so basket detection/cropping stays consistent.
    if len(frames) < 2:
        return frames, {"ecc_success": 0, "flow_success": 0, "identity_fallback": 0}

    ref_frame = frames[0]
    h, w = ref_frame.shape[:2]
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.GaussianBlur(ref_gray, (3, 3), 0)

    stabilised = [ref_frame.copy()]
    ecc_success = 0
    flow_success = 0
    identity_fallback = 0

    for frame in frames[1:]:
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (3, 3), 0)

        warp_matrix = estimate_ecc_euclidean(ref_gray, curr_gray)
        if warp_matrix is not None:
            aligned = cv2.warpAffine(
                frame,
                warp_matrix,
                (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            ecc_success += 1
            stabilised.append(aligned)
            continue

        affine = estimate_affine_with_optical_flow(curr_gray, ref_gray)
        if affine is not None:
            aligned = cv2.warpAffine(
                frame,
                affine,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            flow_success += 1
            stabilised.append(aligned)
            continue

        stabilised.append(frame.copy())
        identity_fallback += 1

    return stabilised, {
        "ecc_success": ecc_success,
        "flow_success": flow_success,
        "identity_fallback": identity_fallback,
    }


def get_crop_bounds(frame_shape, ellipse, padding=12):
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Empty ellipse mask")

    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding + 1, w)
    y1 = max(int(ys.min()) - padding, 0)
    y2 = h
    return x1, y1, x2, y2


def ellipse_in_crop_coords(ellipse, crop_bounds):
    if ellipse is None:
        return None

    x1, y1, _, _ = crop_bounds
    (cx, cy), axes, angle = ellipse
    return (cx - x1, cy - y1), axes, angle


def detect_reference_ellipse(stabilised_frames, manual_roi=False, manual_ellipse=None):
    if len(stabilised_frames) == 0:
        raise RuntimeError("No frames available for detection")

    frame1 = stabilised_frames[0]
    frame2 = stabilised_frames[min(20, len(stabilised_frames) - 1)]

    print("Detecting portafilter ellipse (v2, after stabilisation)...")
    _, ellipse, hole_mode_size, fast_params = detect_elliptical_portafilter_with_holes(
        frame1,
        save_dashboard=False,
        use_interactive=False,
        second_frame=frame2,
        mask_threshold=15,
        manual_roi=manual_roi,
        manual_ellipse=manual_ellipse,
    )

    if ellipse is None:
        raise RuntimeError("Portafilter detection failed")

    print("Portafilter ellipse detected (v2)")
    print(f"Hole mode size: {hole_mode_size}")
    return ellipse, hole_mode_size, fast_params


def save_cropped_frames(stabilised_frames, output_dir, crop_bounds):
    x1, y1, x2, y2 = crop_bounds
    target_h = y2 - y1
    target_w = x2 - x1

    os.makedirs(output_dir, exist_ok=True)

    print("Saving v2 cropped frames")
    for index, frame in enumerate(stabilised_frames):
        roi = frame[y1:y2, x1:x2]
        if roi.shape[:2] != (target_h, target_w):
            roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        output_path = os.path.join(output_dir, f"frame_{index:04d}.jpg")
        cv2.imwrite(output_path, roi)

    print(f"Saved {len(stabilised_frames)} cropped frames")


def process_portafilter_tracking_v2(
    frames_dir=None,
    output_dir=None,
    manual_roi=False,
    manual_ellipse=None,
    stabilise_before_detection=True,
):
    # Stabilise frames first, then detect/crop using the stabilised sequence.
    if frames_dir is None:
        frames_dir = Input_Dir
    if output_dir is None:
        output_dir = Crop_Dir

    frame_files = list_frame_files(frames_dir)
    if not frame_files:
        raise RuntimeError("No frames found in directory")

    frames = load_all_frames(frames_dir, frame_files)
    if len(frames) < 2:
        raise RuntimeError("Not enough frames for tracking")

    if stabilise_before_detection:
        print("Stabilising frames before portafilter detection (v2)")
        stabilised_frames, stabilise_stats = stabilise_frames(frames)
        print(
            "Stabilisation stats: "
            f"ecc={stabilise_stats['ecc_success']}, "
            f"flow={stabilise_stats['flow_success']}, "
            f"fallback={stabilise_stats['identity_fallback']}"
        )
    else:
        stabilised_frames = frames
        stabilise_stats = {"ecc_success": 0, "flow_success": 0, "identity_fallback": 0}

    reference_frame = stabilised_frames[0]
    second_frame = stabilised_frames[min(20, len(stabilised_frames) - 1)]

    if manual_ellipse is not None:
        ellipse = manual_ellipse
        try:
            _, _, hole_mode_size, fast_params = detect_elliptical_portafilter_with_holes(
                reference_frame,
                save_dashboard=False,
                use_interactive=False,
                second_frame=second_frame,
                mask_threshold=15,
                manual_ellipse=ellipse,
            )
        except Exception as err:
            print(f"Manual ROI FAST parameter derivation failed (v2): {err}")
            hole_mode_size = None
            fast_params = {"threshold": 15, "min_circularity": 0.4, "size_tolerance": 5}
    else:
        ellipse, hole_mode_size, fast_params = detect_reference_ellipse(
            stabilised_frames,
            manual_roi=manual_roi,
            manual_ellipse=manual_ellipse,
        )

    crop_bounds = get_crop_bounds(reference_frame.shape, ellipse, padding=12)
    save_cropped_frames(stabilised_frames, output_dir, crop_bounds)
    ellipse_crop_coords = ellipse_in_crop_coords(ellipse, crop_bounds)

    return {
        "ellipse": ellipse,
        "ellipse_in_crop": ellipse_crop_coords,
        "hole_mode_size": hole_mode_size,
        "fast_params": fast_params,
        "output_dir": output_dir,
        "frame_count": len(stabilised_frames),
        "stabilisation": stabilise_stats,
    }


def _clear_image_files(folder):
    if not os.path.isdir(folder):
        return
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                os.remove(path)
            except OSError:
                continue


def run_tracking_v2_on_video(
    video_path,
    frames_dir=None,
    output_dir=None,
    target_fps=1,
    clear_dirs=True,
):
    # Standalone test helper: extract frames from a video, then run v2 tracking.
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if frames_dir is None:
        frames_dir = Input_Dir
    if output_dir is None:
        output_dir = Crop_Dir

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if clear_dirs:
        _clear_image_files(frames_dir)
        _clear_image_files(output_dir)

    print(f"Extracting frames from: {video_path}")
    frame_count = extract_frames(video_path, frames_dir, target_fps=max(1, float(target_fps)))
    print(f"Extracted {frame_count} frames to: {frames_dir}")

    result = process_portafilter_tracking_v2(
        frames_dir=frames_dir,
        output_dir=output_dir,
        manual_roi=False,
        manual_ellipse=None,
        stabilise_before_detection=True,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portafilter tracking v2 (stabilise first)")
    parser.add_argument("--video", type=str, default="", help="Path to input video for modular test")
    parser.add_argument("--frames-dir", type=str, default=Input_Dir, help="Working extracted frames directory")
    parser.add_argument("--output-dir", type=str, default=Crop_Dir, help="Output cropped frames directory")
    parser.add_argument("--target-fps", type=float, default=1.0, help="Frame extraction fps")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing extracted/cropped images")
    args = parser.parse_args()

    if args.video:
        result = run_tracking_v2_on_video(
            video_path=args.video,
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            target_fps=max(1.0, args.target_fps),
            clear_dirs=not args.no_clear,
        )
    else:
        result = process_portafilter_tracking_v2(
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
        )

    print("\n===== PORTAFILTER TRACKING V2 COMPLETE =====")
    print(f"Ellipse: {result['ellipse']}")
    print(f"Hole mode size: {result['hole_mode_size']}")
    print(f"Frames processed: {result['frame_count']}")
