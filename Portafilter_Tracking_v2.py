import os
import cv2
import numpy as np
from Portafilter_Detection import detect_elliptical_portafilter_with_holes, load_image_with_orientation
from Frame_Extraction import (
    DEFAULT_EXTRACTION_FPS,
    PREVIEW_FRAME_OFFSET_SECONDS,
    extract_frames,
    safe_fps,
    sampled_frame_offset_for_seconds,
)


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


def _compute_resize_scale(frame_shape, max_width=None):
    if max_width is None:
        return 1.0
    height, width = frame_shape[:2]
    if width <= 0 or width <= int(max_width):
        return 1.0
    return float(max_width) / float(width)


def _resize_frame(frame, scale):
    if scale >= 0.999:
        return frame
    resized_w = max(1, int(round(frame.shape[1] * float(scale))))
    resized_h = max(1, int(round(frame.shape[0] * float(scale))))
    return cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)


def _scale_ellipse(ellipse, scale):
    if ellipse is None or scale >= 0.999:
        return ellipse
    return (
        (float(ellipse[0][0]) * float(scale), float(ellipse[0][1]) * float(scale)),
        (float(ellipse[1][0]) * float(scale), float(ellipse[1][1]) * float(scale)),
        float(ellipse[2]),
    )


def _rescale_centers(centers, inverse_scale):
    if inverse_scale == 1.0:
        return centers
    return [
        (float(center[0]) * float(inverse_scale), float(center[1]) * float(inverse_scale))
        for center in centers
    ]


def _rescale_fast_params(fast_params, inverse_scale):
    if fast_params is None or inverse_scale == 1.0:
        return fast_params
    scaled = dict(fast_params)
    if "size_tolerance" in scaled:
        try:
            scaled["size_tolerance"] = max(1, int(round(float(scaled["size_tolerance"]) * float(inverse_scale))))
        except (TypeError, ValueError):
            pass
    return scaled


def _rescale_mode_size(hole_mode_size, inverse_scale):
    if hole_mode_size is None or inverse_scale == 1.0:
        return hole_mode_size
    return float(hole_mode_size) * float(inverse_scale)


def preprocess_tracking_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.equalizeHist(gray)


def crop_with_padding(image, bounds):
    x1, y1, x2, y2 = [int(v) for v in bounds]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(image.shape[1], x2)
    src_y2 = min(image.shape[0], y2)

    cropped = image[src_y1:src_y2, src_x1:src_x2]
    if cropped.size == 0:
        if image.ndim == 2:
            return np.zeros((target_h, target_w), dtype=image.dtype)
        return np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)

    pad_left = max(0, src_x1 - x1)
    pad_top = max(0, src_y1 - y1)
    pad_right = max(0, x2 - src_x2)
    pad_bottom = max(0, y2 - src_y2)

    if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
        return cropped

    return cv2.copyMakeBorder(
        cropped,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE,
    )


def get_portafilter_template_bounds(frame_shape, ellipse, padding=18):
    h, w = frame_shape[:2]
    (cx, cy), (major_axis, minor_axis), _ = ellipse
    half_w = max(12, int(round(major_axis / 2.0 + padding)))
    half_h = max(12, int(round(minor_axis / 2.0 + padding)))

    x1 = max(0, int(round(cx)) - half_w)
    x2 = min(w, int(round(cx)) + half_w)
    y1 = max(0, int(round(cy)) - half_h)
    y2 = min(h, int(round(cy)) + half_h)
    return x1, y1, x2, y2


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


def get_centered_bounds(center, width, height):
    cx, cy = center
    x1 = int(round(cx - width / 2.0))
    y1 = int(round(cy - height / 2.0))
    return x1, y1, x1 + int(width), y1 + int(height)


def locate_template_center(
    current_gray,
    template_gray,
    previous_center,
    reference_center,
    frame_shape,
):
    template_h, template_w = template_gray.shape[:2]
    margin_x = max(24, int(round(template_w * 0.9)))
    margin_y = max(24, int(round(template_h * 0.9)))
    search_w = template_w + 2 * margin_x
    search_h = template_h + 2 * margin_y

    best_center = None
    best_score = -1.0
    best_source = "previous"
    seen_sources = set()

    for source_name, anchor in (
        ("previous", previous_center),
        ("reference", reference_center),
    ):
        rounded_anchor = (int(round(anchor[0])), int(round(anchor[1])))
        if rounded_anchor in seen_sources:
            continue
        seen_sources.add(rounded_anchor)

        search_bounds = get_centered_bounds(anchor, search_w, search_h)
        search_patch = crop_with_padding(current_gray, search_bounds)
        if search_patch.shape[0] < template_h or search_patch.shape[1] < template_w:
            continue

        result = cv2.matchTemplate(search_patch, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:
            best_score = float(max_val)
            best_center = (
                float(search_bounds[0] + max_loc[0] + template_w / 2.0),
                float(search_bounds[1] + max_loc[1] + template_h / 2.0),
            )
            best_source = source_name

    if best_center is None:
        best_center = previous_center
        best_score = 0.0
        best_source = "fallback"

    frame_h, frame_w = frame_shape[:2]
    clamped_center = (
        min(max(best_center[0], 0.0), float(frame_w - 1)),
        min(max(best_center[1], 0.0), float(frame_h - 1)),
    )
    return clamped_center, best_score, best_source


def update_filtered_center(previous_center, measured_center, score):
    alpha = float(np.clip((float(score) - 0.35) / 0.45, 0.22, 1.0))
    return (
        previous_center[0] + alpha * (measured_center[0] - previous_center[0]),
        previous_center[1] + alpha * (measured_center[1] - previous_center[1]),
    )


def track_portafilter_motion(frames, ellipse, max_tracking_width=960):
    if len(frames) == 0:
        raise RuntimeError("No frames available for tracking")

    tracking_scale = _compute_resize_scale(frames[0].shape, max_width=max_tracking_width)
    inverse_scale = 1.0 / tracking_scale if tracking_scale > 0 else 1.0
    tracking_frames = [_resize_frame(frame, tracking_scale) for frame in frames]
    ellipse_scaled = _scale_ellipse(ellipse, tracking_scale)
    reference_center = (float(ellipse_scaled[0][0]), float(ellipse_scaled[0][1]))
    tracking_grays = [preprocess_tracking_frame(frame) for frame in tracking_frames]

    template_bounds = get_portafilter_template_bounds(tracking_frames[0].shape, ellipse_scaled, padding=18)
    template_gray = crop_with_padding(tracking_grays[0], template_bounds)
    template_h, template_w = template_gray.shape[:2]

    measured_centers = [reference_center]
    filtered_centers = [reference_center]
    match_scores = [1.0]
    template_updates = 0
    reference_searches = 0
    low_confidence_fallbacks = 0

    for index in range(1, len(frames)):
        measured_center, score, source = locate_template_center(
            tracking_grays[index],
            template_gray,
            filtered_centers[-1],
            reference_center,
            tracking_frames[index].shape,
        )

        if source == "reference":
            reference_searches += 1
        if score < 0.40:
            low_confidence_fallbacks += 1
            measured_center = filtered_centers[-1]

        filtered_center = update_filtered_center(filtered_centers[-1], measured_center, score)

        measured_centers.append(measured_center)
        filtered_centers.append(filtered_center)
        match_scores.append(float(score))

        if score >= 0.60:
            fresh_bounds = get_centered_bounds(measured_center, template_w, template_h)
            fresh_template = crop_with_padding(tracking_grays[index], fresh_bounds)
            fresh_template = _resize_to_shape(fresh_template, template_gray.shape)
            template_gray = cv2.addWeighted(template_gray, 0.85, fresh_template, 0.15, 0.0)
            template_updates += 1

    measured_centers_original = _rescale_centers(measured_centers, inverse_scale)
    filtered_centers_original = _rescale_centers(filtered_centers, inverse_scale)
    reference_center_original = measured_centers_original[0]
    displacements = [
        float(np.hypot(center[0] - reference_center_original[0], center[1] - reference_center_original[1]))
        for center in filtered_centers_original
    ]

    return {
        "reference_center": reference_center_original,
        "measured_centers": measured_centers_original,
        "filtered_centers": filtered_centers_original,
        "match_scores": match_scores,
        "template_shape": (template_h, template_w),
        "template_updates": template_updates,
        "reference_searches": reference_searches,
        "low_confidence_fallbacks": low_confidence_fallbacks,
        "average_match_score": float(np.mean(match_scores)) if match_scores else 0.0,
        "tracking_scale": float(tracking_scale),
        "max_center_displacement": float(max(displacements)) if displacements else 0.0,
    }


def get_locked_crop_bounds(center, reference_crop_bounds, reference_center):
    ref_x1, ref_y1, ref_x2, ref_y2 = reference_crop_bounds
    ref_width = max(1, int(ref_x2 - ref_x1))
    ref_height = max(1, int(ref_y2 - ref_y1))
    left_offset = float(reference_center[0] - ref_x1)
    top_offset = float(reference_center[1] - ref_y1)

    crop_x1 = int(round(center[0] - left_offset))
    crop_y1 = int(round(center[1] - top_offset))
    crop_bounds = (
        crop_x1,
        crop_y1,
        crop_x1 + ref_width,
        crop_y1 + ref_height,
    )
    return crop_bounds


def crop_frame_locked(frame, center, reference_crop_bounds, reference_center):
    crop_bounds = get_locked_crop_bounds(center, reference_crop_bounds, reference_center)
    return crop_with_padding(frame, crop_bounds)


def generate_locked_crops(frames, reference_crop_bounds, reference_center, centers):
    return [
        crop_frame_locked(frame, center, reference_crop_bounds, reference_center)
        for frame, center in zip(frames, centers)
    ]


def _resize_to_shape(image, target_shape):
    target_h, target_w = [int(value) for value in target_shape[:2]]
    image_h, image_w = image.shape[:2]
    if image_h == target_h and image_w == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def save_locked_crops(frames, output_dir, reference_crop_bounds, reference_center, centers):
    os.makedirs(output_dir, exist_ok=True)
    crops = generate_locked_crops(frames, reference_crop_bounds, reference_center, centers)

    print("Saving locked ROI crops (v2)")
    for index, roi in enumerate(crops):
        output_path = os.path.join(output_dir, f"frame_{index:04d}.jpg")
        cv2.imwrite(output_path, roi)

    print(f"Saved {len(crops)} cropped frames")
    return crops


def detect_reference_ellipse(
    frames,
    manual_roi=False,
    manual_ellipse=None,
    detection_max_width=1280,
    sampled_fps=DEFAULT_EXTRACTION_FPS,
):
    if len(frames) == 0:
        raise RuntimeError("No frames available for detection")

    frame1 = frames[0]
    frame2_idx = min(
        sampled_frame_offset_for_seconds(PREVIEW_FRAME_OFFSET_SECONDS, sampled_fps),
        len(frames) - 1,
    )
    frame2 = frames[frame2_idx]
    detection_scale = _compute_resize_scale(frame1.shape, max_width=detection_max_width)
    inverse_scale = 1.0 / detection_scale if detection_scale > 0 else 1.0
    frame1_working = _resize_frame(frame1, detection_scale)
    frame2_working = _resize_frame(frame2, detection_scale)

    print("Detecting portafilter ellipse...")
    _, ellipse, hole_mode_size, fast_params = detect_elliptical_portafilter_with_holes(
        frame1_working,
        save_dashboard=False,
        use_interactive=False,
        second_frame=frame2_working,
        mask_threshold=15,
        manual_roi=manual_roi,
        manual_ellipse=manual_ellipse,
    )

    if ellipse is None:
        raise RuntimeError("Portafilter detection failed")

    print("Portafilter ellipse detected")
    print(f"Hole mode size: {hole_mode_size}")
    return (
        _scale_ellipse(ellipse, inverse_scale),
        _rescale_mode_size(hole_mode_size, inverse_scale),
        _rescale_fast_params(fast_params, inverse_scale),
    )


def process_portafilter_tracking_v2(
    frames_dir=None,
    output_dir=None,
    frames=None,
    manual_roi=False,
    manual_ellipse=None,
    stabilise_before_detection=True,
    save_crops=True,
    return_crops=False,
    detection_max_width=1280,
    tracking_max_width=960,
    sampled_fps=DEFAULT_EXTRACTION_FPS,
):
    if frames_dir is None:
        frames_dir = Input_Dir
    if output_dir is None:
        output_dir = Crop_Dir

    sampled_fps = safe_fps(sampled_fps)
    if frames is None:
        frame_files = list_frame_files(frames_dir)
        if not frame_files:
            raise RuntimeError("No frames found in directory")
        frames = load_all_frames(frames_dir, frame_files)
    else:
        frames = list(frames)

    if len(frames) < 2:
        raise RuntimeError("Not enough frames for tracking")

    reference_frame = frames[0]
    second_frame_idx = min(
        sampled_frame_offset_for_seconds(PREVIEW_FRAME_OFFSET_SECONDS, sampled_fps),
        len(frames) - 1,
    )
    second_frame = frames[second_frame_idx]

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
            frames,
            manual_roi=manual_roi,
            manual_ellipse=manual_ellipse,
            detection_max_width=detection_max_width,
            sampled_fps=sampled_fps,
        )

    crop_bounds = get_crop_bounds(reference_frame.shape, ellipse, padding=12)
    reference_center = (float(ellipse[0][0]), float(ellipse[0][1]))

    if stabilise_before_detection:
        print("Tracking basket motion and locking the ROI (v2)")
        tracking = track_portafilter_motion(frames, ellipse, max_tracking_width=tracking_max_width)
        crop_centers = tracking["filtered_centers"]
        print(
            "Tracking stats: "
            f"avg_score={tracking['average_match_score']:.3f}, "
            f"template_updates={tracking['template_updates']}, "
            f"reference_searches={tracking['reference_searches']}, "
            f"fallbacks={tracking['low_confidence_fallbacks']}"
        )
    else:
        tracking = {
            "reference_center": reference_center,
            "measured_centers": [reference_center for _ in frames],
            "filtered_centers": [reference_center for _ in frames],
            "match_scores": [1.0 for _ in frames],
            "template_shape": None,
            "template_updates": 0,
            "reference_searches": 0,
            "low_confidence_fallbacks": 0,
            "average_match_score": 1.0,
            "max_center_displacement": 0.0,
        }
        crop_centers = tracking["filtered_centers"]

    cropped_frames = None
    if save_crops:
        cropped_frames = save_locked_crops(
            frames,
            output_dir,
            reference_crop_bounds=crop_bounds,
            reference_center=reference_center,
            centers=crop_centers,
        )
    elif return_crops:
        cropped_frames = generate_locked_crops(
            frames,
            reference_crop_bounds=crop_bounds,
            reference_center=reference_center,
            centers=crop_centers,
        )

    ellipse_crop_coords = ellipse_in_crop_coords(ellipse, crop_bounds)

    result = {
        "ellipse": ellipse,
        "ellipse_in_crop": ellipse_crop_coords,
        "hole_mode_size": hole_mode_size,
        "fast_params": fast_params,
        "output_dir": output_dir,
        "frame_count": len(frames),
        "fps": sampled_fps,
        "crop_bounds": tuple(int(value) for value in crop_bounds),
        "reference_center": tuple(float(value) for value in reference_center),
        "crop_centers": [
            (float(center[0]), float(center[1]))
            for center in crop_centers
        ],
        "stabilisation": {
            "method": "roi_template_lock",
            "average_match_score": tracking["average_match_score"],
            "template_updates": tracking["template_updates"],
            "reference_searches": tracking["reference_searches"],
            "low_confidence_fallbacks": tracking["low_confidence_fallbacks"],
            "max_center_displacement": tracking["max_center_displacement"],
        },
    }
    if return_crops:
        result["cropped_frames"] = cropped_frames if cropped_frames is not None else []
    return result


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
    target_fps=DEFAULT_EXTRACTION_FPS,
    clear_dirs=True,
):
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
    target_fps = safe_fps(target_fps)
    frame_count = extract_frames(video_path, frames_dir, target_fps=target_fps)
    print(f"Extracted {frame_count} frames to: {frames_dir}")

    result = process_portafilter_tracking_v2(
        frames_dir=frames_dir,
        output_dir=output_dir,
        manual_roi=False,
        manual_ellipse=None,
        stabilise_before_detection=True,
        sampled_fps=target_fps,
    )
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portafilter tracking v2 (ROI lock)")
    parser.add_argument("--video", type=str, default="", help="Path to input video for modular test")
    parser.add_argument("--frames-dir", type=str, default=Input_Dir, help="Working extracted frames directory")
    parser.add_argument("--output-dir", type=str, default=Crop_Dir, help="Output cropped frames directory")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_EXTRACTION_FPS, help="Frame extraction fps")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing extracted/cropped images")
    args = parser.parse_args()

    if args.video:
        result = run_tracking_v2_on_video(
            video_path=args.video,
            frames_dir=args.frames_dir,
            output_dir=args.output_dir,
            target_fps=safe_fps(args.target_fps),
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
