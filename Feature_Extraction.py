import os
import cv2
import numpy as np
import json
from collections import deque
from typing import Dict, Tuple, List
from Portafilter_Detection import detect_elliptical_portafilter_with_holes

# Helpers

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Crop_Dir = os.path.join(Base_Dir, "Image Data", "Cropped")
Output_Dir = os.path.join(Base_Dir, "Analysis")
os.makedirs(Output_Dir, exist_ok=True)


def filter_keypoints_by_size(keypoints, keypoint_sizes, target_size, tolerance=5):
    if target_size is None:
        return keypoints

    filtered_keypoints = []
    for keypoint, size in zip(keypoints, keypoint_sizes):
        if abs(size - target_size) <= tolerance:
            filtered_keypoints.append(keypoint)
    return filtered_keypoints


def detect_fast_circles(image, threshold=25, min_circularity=0.4):
    if image is None or image.size == 0:
        return [], []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    try:
        fast_create = getattr(cv2, "FastFeatureDetector_create", None)
        if fast_create is not None:
            fast = fast_create(threshold=threshold)
        else:
            fast_class = getattr(cv2, "FastFeatureDetector", None)
            if fast_class is not None:
                fast = fast_class.create(threshold=threshold)
            else:
                raise RuntimeError("No FAST detector available in this OpenCV build")

        keypoints = fast.detect(gray, None)
    except Exception:
        return [], []

    if not keypoints:
        return [], []

    filtered_keypoints = []
    keypoint_sizes = []
    height, width = gray.shape[:2]

    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        size = int(getattr(keypoint, "size", 20))
        half_size = max(1, size // 2)

        if x < half_size or y < half_size or x >= width - half_size or y >= height - half_size:
            continue

        x1, y1 = max(0, x - half_size), max(0, y - half_size)
        x2, y2 = min(width, x + half_size), min(height, y + half_size)
        region = gray[y1:y2, x1:x2]

        if region.size == 0:
            continue

        try:
            _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area <= 0:
                continue

            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue

            filtered_keypoints.append(keypoint)
            keypoint_sizes.append(size)
        except Exception:
            # Keep prior behavior: skip points that fail contour validation.
            continue

    return filtered_keypoints, keypoint_sizes

def load_frames(folder):
    frame_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    frames_gray = []
    frames = []

    for fname in frame_files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frames.append(img)
        frames_gray.append(gray)

    if len(frames_gray) == 0:
        raise RuntimeError("No frames loaded from stabilised directory")

    return frames, frames_gray, frame_files

def detect_flow_start_end(
    frames_gray: List[np.ndarray],
    roi: Tuple[int, int, int, int],
    pixel_diff_thresh: int = 15,
    change_fraction_thresh: float = 0.005,
    start_consec_frames: int = 2,
    end_consec_frames: int = 5,
    smoothing_window: int = 3,
    end_lookback_window: int = 10
) -> Tuple[int, int]:
    # Detect the shot start/end window from frame-to-frame changes.
    # ROI tuple order here is (y1, y2, x1, x2).
    y1, y2, x1, x2 = roi
    roi_area = (y2 - y1) * (x2 - x1)

    if roi_area <= 0:
        raise ValueError("Invalid ROI dimensions")

    change_fractions = []

    # Calculate frame-to-frame changes
    for i in range(1, len(frames_gray)):
        prev = frames_gray[i - 1][y1:y2, x1:x2].astype(np.int16)
        curr = frames_gray[i][y1:y2, x1:x2].astype(np.int16)

        diff = np.abs(curr - prev)
        mask = (diff > pixel_diff_thresh).astype(np.uint8) * 255

        change_fraction = np.sum(mask > 0) / roi_area

        change_fractions.append(change_fraction)

    # Smooth the change curve to reduce noise
    change_fractions_array = np.array(change_fractions)
    if len(change_fractions_array) >= smoothing_window:
        smoothed_changes = np.convolve(
            change_fractions_array, 
            np.ones(smoothing_window) / smoothing_window, 
            mode='same'
        )
    else:
        smoothed_changes = change_fractions_array

    # Detect flow start using smoothed curve
    start_frame = None
    consec = 0

    for i, frac in enumerate(smoothed_changes):
        if frac >= change_fraction_thresh:
            consec += 1
            if consec >= start_consec_frames:
                # Go back to find the actual start (before smoothing effect)
                start_frame = max(0, i - start_consec_frames + 1)
                break
        else:
            consec = 0

    if start_frame is None:
        print("Warning: no flow start detected, using frame 0")
        return 0, len(frames_gray) - 1

    # Detect flow end - use a more sophisticated approach for choked shots
    # Look for sustained low activity rather than immediate cutoff
    end_frame = len(frames_gray) - 1  # fallback to last frame
    
    # Calculate a rolling average to smooth out temporary dips
    window_size = min(end_lookback_window, len(smoothed_changes) - start_frame)
    
    if window_size >= 3:
        rolling_avg = []
        for i in range(start_frame, len(smoothed_changes)):
            window_start = max(start_frame, i - window_size + 1)
            window_data = smoothed_changes[window_start:i+1]
            rolling_avg.append(np.mean(window_data))
        
        # Find where rolling average drops below threshold for extended period
        consec = 0
        for i, avg in enumerate(rolling_avg):
            actual_idx = start_frame + i
            if avg < change_fraction_thresh * 0.5:  # Lower threshold for end detection
                consec += 1
                if consec >= end_consec_frames:
                    end_frame = actual_idx - consec + 1
                    break
            else:
                consec = 0
    else:
        # Fallback to simple consecutive frame counting
        consec = 0
        for i in range(start_frame, len(smoothed_changes)):
            if smoothed_changes[i] < change_fraction_thresh * 0.5:
                consec += 1
                if consec >= end_consec_frames:
                    end_frame = i - consec + 1
                    break
            else:
                consec = 0

    end_frame = max(end_frame, start_frame)
    
    print(f"Flow detection: start={start_frame}, end={end_frame}, duration={end_frame - start_frame + 1} frames")
    print(f"Peak change fraction: {np.max(smoothed_changes[start_frame:end_frame+1]):.4f}")
    print(f"Mean change fraction during shot: {np.mean(smoothed_changes[start_frame:end_frame+1]):.4f}")

    return start_frame, end_frame


def get_analysis_roi_from_ellipse(ellipse, frame_shape, height_multiplier=2.0):
    # Build a rectangle from the basket ellipse, then extend downward for the stream.
    if ellipse is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    x2 = min(int(xs.max()) + 1, frame_w)
    y1 = int(ys.min())
    y2 = min(int(ys.max()) + 1, frame_h)

    ellipse_h = y2 - y1
    if ellipse_h <= 0:
        return None

    y2 = min(frame_h, y1 + int(ellipse_h * height_multiplier))

    return x1, y1, x2, y2


def get_triangle_points_from_roi(roi, frame_shape):
    # Triangle vertices: left-top, right-top, and bottom-center of ROI rectangle.
    if roi is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

    if x1 >= x2 or y1 >= y2:
        return None

    top_left = (x1, y1)
    top_right = (x2 - 1, y1)
    bottom_center = ((x1 + x2) // 2, y2 - 1)
    return np.array([top_left, top_right, bottom_center], dtype=np.int32)


def get_bottom_triangle_points_from_roi(roi, frame_shape, keep_fraction=2.0 / 3.0):
    # Keep only the lower part of the triangle to focus on the stream path.
    points = get_triangle_points_from_roi(roi, frame_shape)
    if points is None:
        return None

    top_left = points[0].astype(np.float32)
    top_right = points[1].astype(np.float32)
    bottom = points[2].astype(np.float32)

    trim_fraction = 1.0 - float(keep_fraction)
    trim_fraction = min(1.0, max(0.0, trim_fraction))
    left_cut = top_left + trim_fraction * (bottom - top_left)
    right_cut = top_right + trim_fraction * (bottom - top_right)

    clipped = np.array(
        [
            (int(round(left_cut[0])), int(round(left_cut[1]))),
            (int(round(right_cut[0])), int(round(right_cut[1]))),
            (int(round(bottom[0])), int(round(bottom[1]))),
        ],
        dtype=np.int32,
    )
    return clipped


def make_bottom_triangle_mask(frame_shape, roi):
    if roi is None:
        return None

    points = get_bottom_triangle_points_from_roi(roi, frame_shape, keep_fraction=2.0 / 3.0)
    if points is None:
        return None

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, points, 255)
    return mask


def make_channeling_roi_mask(frame_shape, portafilter_ellipse=None):
    # Channeling ROI uses only the basket ellipse.
    if portafilter_ellipse is None:
        return None

    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    try:
        cv2.ellipse(mask, portafilter_ellipse, 255, -1)
    except Exception:
        return None
    return mask


def make_blonding_roi_mask(frame_shape, analysis_rect=None, portafilter_ellipse=None):
    # Blonding ROI is union: ellipse + bottom two-thirds of triangle.
    union_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    has_region = False

    triangle_mask = make_bottom_triangle_mask(frame_shape, analysis_rect)
    if triangle_mask is not None:
        union_mask = cv2.bitwise_or(union_mask, triangle_mask)
        has_region = True

    if portafilter_ellipse is not None:
        ellipse_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        try:
            cv2.ellipse(ellipse_mask, portafilter_ellipse, 255, -1)
            union_mask = cv2.bitwise_or(union_mask, ellipse_mask)
            has_region = True
        except Exception:
            pass

    if not has_region:
        return None
    return union_mask


def get_stream_width_roi_from_ellipse(
    portafilter_ellipse,
    frame_shape,
    width_fraction=0.72,
    bottom_anchor_fraction=0.82,
):
    # Width/veer analysis focuses on a narrow vertical corridor below the basket outlet.
    if portafilter_ellipse is None:
        return None

    frame_h, frame_w = frame_shape[:2]
    cx, cy = portafilter_ellipse[0]
    basket_w, basket_h = portafilter_ellipse[1]

    roi_w = max(16, int(float(basket_w) * float(width_fraction)))
    x1 = max(0, int(round(cx - roi_w / 2.0)))
    x2 = min(frame_w, int(round(cx + roi_w / 2.0)))

    outlet_y = float(cy) + float(basket_h) * 0.12
    y1 = max(0, min(frame_h - 2, int(round(outlet_y))))
    y2 = min(frame_h, int(round(frame_h * float(bottom_anchor_fraction))))
    if y2 <= y1:
        y2 = min(frame_h, y1 + max(20, frame_h // 4))

    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    return x1, y1, x2, y2


def _find_row_segments(binary_row):
    cols = np.where(binary_row > 0)[0]
    if len(cols) == 0:
        return []

    split_points = np.where(np.diff(cols) > 1)[0]
    starts = np.concatenate(([0], split_points + 1))
    ends = np.concatenate((split_points, [len(cols) - 1]))
    return [(int(cols[start]), int(cols[end])) for start, end in zip(starts, ends)]


def _make_stream_seed_mask(mask_shape):
    roi_h, roi_w = mask_shape[:2]
    seed_h = max(6, int(round(roi_h * 0.16)))
    seed_half_w = max(5, int(round(roi_w * 0.14)))
    seed_cx = roi_w // 2

    seed_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
    sx1 = max(0, seed_cx - seed_half_w)
    sx2 = min(roi_w, seed_cx + seed_half_w)
    seed_mask[:seed_h, sx1:sx2] = 255
    return seed_mask


def _build_stream_candidate_mask(roi_bgr, roi_gray, roi_stream=None):
    roi_h, roi_w = roi_gray.shape[:2]
    if roi_h < 2 or roi_w < 2:
        return None

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray_float = roi_gray.astype(np.float32)

    row_background = np.percentile(gray_float, 60, axis=1, keepdims=True)
    darkness_delta = row_background - gray_float
    dark_relative_mask = darkness_delta >= max(7.0, float(np.percentile(darkness_delta, 82)))

    kernel_w = max(3, int(round(roi_w * 0.08)))
    kernel_h = max(9, int(round(roi_h * 0.20)))
    if kernel_w % 2 == 0:
        kernel_w += 1
    if kernel_h % 2 == 0:
        kernel_h += 1
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    blackhat = cv2.morphologyEx(roi_gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
    blackhat_mask = blackhat >= max(10, int(np.percentile(blackhat, 84)))

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    warm_mask = (
        (hue >= 6.0)
        & (hue <= 34.0)
        & (sat >= max(18.0, np.percentile(sat, 25)))
        & (val >= 8.0)
        & (val <= max(210.0, np.percentile(val, 88)))
    )

    candidate = (dark_relative_mask & blackhat_mask) | (dark_relative_mask & warm_mask)
    if roi_stream is not None:
        motion_hint = roi_stream > 0
        candidate = candidate | ((dark_relative_mask | warm_mask) & motion_hint)

    candidate_mask = candidate.astype(np.uint8) * 255
    if np.count_nonzero(candidate_mask) < 12:
        candidate_mask = (dark_relative_mask.astype(np.uint8) * 255)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(7, kernel_h // 2)))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, vertical_kernel)
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    return candidate_mask


def _select_seeded_stream_component(candidate_mask, seed_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    if num_labels <= 1:
        return None

    roi_h, roi_w = candidate_mask.shape[:2]
    roi_center_x = (roi_w - 1) / 2.0
    seed_binary = seed_mask > 0
    best_component = None
    best_score = -1e9

    for label in range(1, num_labels):
        component_mask = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 10:
            continue

        ys, xs = np.where(component_mask)
        if len(xs) == 0 or len(ys) == 0:
            continue

        top = int(np.min(ys))
        bottom = int(np.max(ys))
        left = int(np.min(xs))
        right = int(np.max(xs))
        height = bottom - top + 1
        width = right - left + 1
        bbox_area = max(1, height * width)
        fill_ratio = float(area) / float(bbox_area)
        aspect_ratio = float(height) / float(max(1, width))
        center_x = float(np.median(xs))
        center_offset = abs(center_x - roi_center_x) / max(1.0, roi_w / 2.0)
        seed_overlap = float(np.count_nonzero(component_mask & seed_binary))
        touches_origin = bool(seed_overlap > 0 or top <= max(4, seed_mask.shape[0] // 2))
        vertical_coverage = float(height) / float(max(1, roi_h))

        score = (
            seed_overlap * 0.35
            + (3.0 if touches_origin else 0.0)
            + min(aspect_ratio, 8.0) * 1.2
            + vertical_coverage * 3.0
            - center_offset * 2.5
            - max(0.0, fill_ratio - 0.45) * 2.0
        )

        if best_component is None or score > best_score:
            best_component = component_mask
            best_score = score

    if best_component is None:
        return None

    refined_mask = np.zeros_like(candidate_mask, dtype=np.uint8)
    refined_mask[best_component] = 255
    refined_mask = cv2.morphologyEx(
        refined_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)),
    )
    return refined_mask


def _classify_stream_veer(offset_px, slope_px_per_row, roi_width):
    offset_norm = abs(float(offset_px)) / max(1.0, float(roi_width) * 0.10)
    slope_norm = abs(float(slope_px_per_row)) / max(1.0, float(roi_width) * 0.02)

    if abs(float(offset_px)) <= max(2.0, float(roi_width) * 0.03) and abs(float(slope_px_per_row)) <= 0.05:
        return "center", 0.0

    direction = "right" if float(offset_px) > 0 else "left"
    magnitude = float(max(offset_norm, slope_norm))
    return direction, magnitude


def extract_stream_component_in_roi(
    frame_bgr,
    frame_gray,
    stream_mask,
    width_roi,
):
    # Refine one outlet-anchored stream component from direct appearance cues.
    if width_roi is None:
        return None

    x1, y1, x2, y2 = width_roi
    if x1 >= x2 or y1 >= y2:
        return None

    roi_bgr = frame_bgr[y1:y2, x1:x2]
    roi_gray = frame_gray[y1:y2, x1:x2]
    roi_stream = stream_mask[y1:y2, x1:x2]
    if roi_bgr.size == 0 or roi_gray.size == 0 or roi_stream.size == 0:
        return None

    candidate_mask = _build_stream_candidate_mask(roi_bgr, roi_gray, roi_stream=roi_stream)
    if candidate_mask is None:
        return None
    seed_mask = _make_stream_seed_mask(candidate_mask.shape)
    return _select_seeded_stream_component(candidate_mask, seed_mask)


def detect_stream_width_in_roi(
    frame_bgr,
    frame_gray,
    stream_mask,
    width_roi,
    min_rows=6,
    measurement_row_fraction=0.70,
):
    # Measure width and veer from an outlet-anchored centerline over many rows.
    if width_roi is None:
        return None

    x1, y1, x2, y2 = width_roi
    if x1 >= x2 or y1 >= y2:
        return None

    refined_mask = extract_stream_component_in_roi(
        frame_bgr,
        frame_gray,
        stream_mask,
        width_roi,
    )
    if refined_mask is None or np.count_nonzero(refined_mask) < 12:
        return None

    roi_h, roi_w = refined_mask.shape[:2]
    if roi_h < 2 or roi_w < 2:
        return None

    outlet_center_x = (roi_w - 1) / 2.0
    row_positions = []
    left_cols = []
    right_cols = []
    widths = []
    centers = []

    for row_idx in range(roi_h):
        cols = np.where(refined_mask[row_idx] > 0)[0]
        if len(cols) < 2:
            continue

        left = float(np.percentile(cols, 12))
        right = float(np.percentile(cols, 88))
        width = right - left + 1.0
        if width < 2.0:
            continue
        center = float(np.median(cols))

        row_positions.append(float(row_idx))
        left_cols.append(left)
        right_cols.append(right)
        widths.append(float(width))
        centers.append(center)

    if len(row_positions) < min_rows:
        return None

    rows = np.array(row_positions, dtype=np.float32)
    left_cols = np.array(left_cols, dtype=np.float32)
    right_cols = np.array(right_cols, dtype=np.float32)
    widths = np.array(widths, dtype=np.float32)
    centers = np.array(centers, dtype=np.float32)

    if len(rows) >= 8:
        fit_start = int(len(rows) * 0.18)
        fit_end = max(fit_start + 3, int(len(rows) * 0.90))
        fit_slice = slice(fit_start, fit_end)
    else:
        fit_slice = slice(0, len(rows))

    fit_rows = rows[fit_slice]
    fit_centers = centers[fit_slice]
    fit_widths = widths[fit_slice]
    if len(fit_rows) < min_rows:
        fit_rows = rows
        fit_centers = centers
        fit_widths = widths

    line_slope = 0.0
    line_intercept = float(np.median(fit_centers))
    if len(fit_rows) >= 2:
        line_slope, line_intercept = np.polyfit(fit_rows, fit_centers, 1)

    fitted_centers = line_slope * fit_rows + line_intercept
    residual_std = float(np.std(fit_centers - fitted_centers)) if len(fit_centers) > 1 else 0.0
    width_px = float(np.median(fit_widths))
    width_std = float(np.std(fit_widths)) if len(fit_widths) > 1 else 0.0

    target_row = float(np.quantile(fit_rows, measurement_row_fraction))
    measurement_idx = int(np.argmin(np.abs(rows - target_row)))
    measurement_row = int(round(rows[measurement_idx]))
    left = int(round(left_cols[measurement_idx]))
    right = int(round(right_cols[measurement_idx]))
    center_px = float((left + right) / 2.0)

    bottom_row = float(np.quantile(fit_rows, 0.90))
    bottom_center = float(line_slope * bottom_row + line_intercept)
    center_offset_px = float(bottom_center - outlet_center_x)
    centeredness_score = float(abs(center_offset_px))
    straightness_score = float(residual_std + abs(float(line_slope)) * 3.0 + width_std * 0.18)
    veer_direction, veer_score = _classify_stream_veer(center_offset_px, line_slope, roi_w)

    overlay_step = max(1, len(rows) // 18)
    left_edges = [
        (x1 + int(round(left_cols[idx])), y1 + int(round(rows[idx])))
        for idx in range(0, len(rows), overlay_step)
    ]
    right_edges = [
        (x1 + int(round(right_cols[idx])), y1 + int(round(rows[idx])))
        for idx in range(0, len(rows), overlay_step)
    ]
    center_line = [
        (x1 + int(round(centers[idx])), y1 + int(round(rows[idx])))
        for idx in range(0, len(rows), overlay_step)
    ]
    fit_line = [
        (x1 + int(round(line_slope * row + line_intercept)), y1 + int(round(row)))
        for row in (fit_rows[0], fit_rows[-1])
    ]

    return {
        "width_px": width_px,
        "width_median_px": width_px,
        "width_std_px": width_std,
        "best_row": measurement_row,
        "row_count": int(len(rows)),
        "center_px": float(center_px),
        "center_offset_px": center_offset_px,
        "centeredness_score": float(centeredness_score),
        "center_std": residual_std,
        "center_slope": float(line_slope),
        "vertical_line_score": float(abs(line_slope)),
        "edge_strength": float(max(0.0, 255.0 - residual_std * 40.0)),
        "measurement_row_fraction": float(measurement_row / max(1, roi_h - 1)),
        "measurement_row_y": int(y1 + measurement_row),
        "straightness_score": straightness_score,
        "veer_direction": veer_direction,
        "veer_score": float(veer_score),
        "left_edges": left_edges,
        "right_edges": right_edges,
        "center_line": center_line,
        "fit_line": fit_line,
        "refined_component_mask": refined_mask,
    }


def draw_stream_width_overlay(frame, width_roi=None, width_measurement=None):
    if width_roi is not None:
        x1, y1, x2, y2 = width_roi
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (255, 180, 0), 2)

    if width_measurement is None:
        return frame

    left_edges = width_measurement.get("left_edges") or []
    right_edges = width_measurement.get("right_edges") or []
    edge_color = (255, 0, 0)
    row_color = (80, 80, 255)
    center_color = (0, 255, 0)
    fit_color = (0, 200, 255)

    measurement_row_y = width_measurement.get("measurement_row_y")
    if width_roi is not None and measurement_row_y is not None:
        x1, _, x2, _ = width_roi
        y = int(measurement_row_y)
        cv2.line(frame, (x1, y), (x2 - 1, y), row_color, 1)

    if len(left_edges) > 1:
        cv2.polylines(frame, [np.array(left_edges, dtype=np.int32)], isClosed=False, color=edge_color, thickness=1)
    if len(right_edges) > 1:
        cv2.polylines(frame, [np.array(right_edges, dtype=np.int32)], isClosed=False, color=edge_color, thickness=1)
    center_line = width_measurement.get("center_line") or []
    if len(center_line) > 1:
        cv2.polylines(frame, [np.array(center_line, dtype=np.int32)], isClosed=False, color=center_color, thickness=1)
    fit_line = width_measurement.get("fit_line") or []
    if len(fit_line) > 1:
        cv2.line(frame, fit_line[0], fit_line[-1], fit_color, 1)

    if len(left_edges) == 1 and len(right_edges) == 1:
        cv2.line(frame, left_edges[0], right_edges[0], edge_color, 2)

    for point in left_edges:
        cv2.circle(frame, point, 2, edge_color, -1)
    for point in right_edges:
        cv2.circle(frame, point, 2, edge_color, -1)

    return frame


def render_stream_detection_preview(
    frame_bgr,
    analysis_rect=None,
    portafilter_ellipse=None,
    stream_width_roi=None,
    stream_mask=None,
    width_measurement=None,
):
    # Desktop UI preview should reflect the current outlet-anchored detector, not the accumulated blonding mask.
    preview = cv2.convertScaleAbs(frame_bgr, alpha=0.48, beta=0)

    if stream_mask is not None and np.count_nonzero(stream_mask) > 0:
        raw_overlay = np.zeros_like(preview)
        raw_overlay[:, :, 0] = np.maximum(raw_overlay[:, :, 0], stream_mask)
        raw_overlay[:, :, 1] = np.maximum(raw_overlay[:, :, 1], (stream_mask // 3))
        preview = cv2.addWeighted(preview, 1.0, raw_overlay, 0.30, 0.0)

    if width_measurement is not None and stream_width_roi is not None:
        refined_component = width_measurement.get("refined_component_mask")
        if refined_component is not None and np.count_nonzero(refined_component) > 0:
            x1, y1, x2, y2 = stream_width_roi
            refined_overlay = np.zeros_like(preview)
            refined_overlay[y1:y2, x1:x2, 1] = np.maximum(
                refined_overlay[y1:y2, x1:x2, 1],
                refined_component,
            )
            preview = cv2.addWeighted(preview, 1.0, refined_overlay, 0.60, 0.0)

    draw_analysis_roi_on_frame(
        preview,
        analysis_rect,
        portafilter_ellipse=portafilter_ellipse,
    )
    draw_stream_width_overlay(
        preview,
        width_roi=stream_width_roi,
        width_measurement=width_measurement,
    )

    legend_y = 22
    cv2.putText(
        preview,
        "Blue: broad stream mask",
        (12, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 220, 120),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "Green: selected stream path",
        (12, legend_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 255, 120),
        1,
        cv2.LINE_AA,
    )

    if width_measurement is not None:
        veer_direction = str(width_measurement.get("veer_direction", "center"))
        veer_score = float(width_measurement.get("veer_score", 0.0))
        width_px = float(width_measurement.get("width_px", 0.0))
        metric_text = f"Width {width_px:.1f}px | Veer {veer_direction} ({veer_score:.2f})"
        cv2.putText(
            preview,
            metric_text,
            (12, legend_y + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    return preview

def detect_channeling_in_frame(
    frame_gray,
    portafilter_ellipse=None,
    target_hole_size=None,
    fast_params=None,
    analysis_rect=None,
):
    # Count visible holes inside the portafilter ellipse region.
    # Use FAST parameters from portafilter detection to ensure consistency
    if fast_params is None:
        fast_params = {'threshold': 15, 'min_circularity': 0.4, 'size_tolerance': 5}
    
    frame_h, frame_w = frame_gray.shape
    # Channeling intentionally ignores the triangle and stays ellipse-only.
    roi_mask = make_channeling_roi_mask(
        (frame_h, frame_w),
        portafilter_ellipse=portafilter_ellipse,
    )

    if roi_mask is not None:
        # Apply mask to image - zeros out everything outside ROI
        masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=roi_mask)
    else:
        masked_frame = frame_gray
    
    # Detect circular features (holes) using exact FAST params from portafilter detection.
    keypoints, keypoint_sizes = detect_fast_circles(
        masked_frame, 
        threshold=fast_params['threshold'],
        min_circularity=fast_params['min_circularity']
    )
    
    # Filter keypoints to only those inside the ROI.
    if roi_mask is not None and len(keypoints) > 0:
        filtered_by_roi = []
        filtered_sizes_by_roi = []

        for kp, size in zip(keypoints, keypoint_sizes):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # Keep only keypoints that are inside the analysis ROI
            if 0 <= y < frame_h and 0 <= x < frame_w and roi_mask[y, x] == 255:
                filtered_by_roi.append(kp)
                filtered_sizes_by_roi.append(size)
        
        keypoints = filtered_by_roi
        keypoint_sizes = filtered_sizes_by_roi
    
    # Filter to target hole size using the same tolerance from portafilter detection
    if target_hole_size is not None and len(keypoints) > 0:
        filtered_keypoints = filter_keypoints_by_size(
            keypoints, 
            keypoint_sizes, 
            target_hole_size, 
            tolerance=fast_params['size_tolerance']
        )
        if len(filtered_keypoints) == 0:
            # Keep the overlay usable even when detected point sizes drift briefly.
            filtered_keypoints = keypoints
    else:
        filtered_keypoints = keypoints
    
    return len(filtered_keypoints), filtered_keypoints


def compute_stream_mask(prev_gray,
                        curr_gray,
                        curr_bgr,
                        diff_thresh=25,
                        brightness_thresh=130,
                        min_component_area=30,
                        motion_fraction_thresh=0.3,
                        exclude_bottom_fraction=0.0,
                        analysis_rect=None,
                        portafilter_ellipse=None,
                        stream_width_roi=None):
    # Stream mask tuned for both dense cone and thin downstream stream.

    H, W = curr_gray.shape

    diff = cv2.absdiff(curr_gray, prev_gray)
    motion_mask_strong = diff > diff_thresh
    motion_mask_soft = diff > max(6, int(diff_thresh * 0.35))

    hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    Hc = hsv[:, :, 0]
    Sc = hsv[:, :, 1]
    V = hsv[:, :, 2]
    dark_mask = V < brightness_thresh

    warm_mask = (Hc >= 8) & (Hc <= 34) & (Sc >= 55) & (V >= 18) & (V <= 220)
    brown_mask = (Hc >= 5) & (Hc <= 26) & (Sc >= 30) & (V >= 10) & (V <= 180)
    color_mask = warm_mask | brown_mask

    combined_strict = dark_mask & motion_mask_strong
    combined_soft_color_motion = color_mask & motion_mask_soft

    width_roi_mask = None
    if stream_width_roi is not None:
        x1, y1, x2, y2 = stream_width_roi
        x1 = max(0, min(W - 1, int(x1)))
        x2 = max(0, min(W, int(x2)))
        y1 = max(0, min(H - 1, int(y1)))
        y2 = max(0, min(H, int(y2)))
        if x2 > x1 and y2 > y1:
            width_roi_mask = np.zeros((H, W), dtype=np.uint8)
            width_roi_mask[y1:y2, x1:x2] = 255

    # Build an adaptive color gate from the stream origin band to better match each shot.
    adaptive_color_mask = np.zeros((H, W), dtype=bool)
    seed_mask = None
    seed_y_max = None
    width_roi_area = 0
    width_roi_center_x = W / 2.0
    width_roi_width = float(W)
    if width_roi_mask is not None:
        width_roi_area = int(np.count_nonzero(width_roi_mask))
        x_idx, y_idx = np.where(width_roi_mask > 0)[1], np.where(width_roi_mask > 0)[0]
        if len(x_idx) > 0 and len(y_idx) > 0:
            x1r = int(np.min(x_idx))
            x2r = int(np.max(x_idx)) + 1
            y1r = int(np.min(y_idx))
            y2r = int(np.max(y_idx)) + 1
            width_roi_center_x = (x1r + x2r) / 2.0
            width_roi_width = float(max(1, x2r - x1r))

            seed_h = max(8, int((y2r - y1r) * 0.16))
            seed_half_w = max(6, int((x2r - x1r) * 0.22))
            sx1 = max(0, int(round(width_roi_center_x)) - seed_half_w)
            sx2 = min(W, int(round(width_roi_center_x)) + seed_half_w)
            sy1 = y1r
            sy2 = min(H, y1r + seed_h)

            seed_mask = np.zeros((H, W), dtype=np.uint8)
            if sx2 > sx1 and sy2 > sy1:
                seed_mask[sy1:sy2, sx1:sx2] = 255
                seed_y_max = sy2 - 1

                seed_selector = (seed_mask > 0) & motion_mask_soft & ((dark_mask) | (color_mask))
                if np.count_nonzero(seed_selector) >= 25:
                    seed_h_vals = Hc[seed_selector].astype(np.float32)
                    seed_s_vals = Sc[seed_selector].astype(np.float32)
                    seed_v_vals = V[seed_selector].astype(np.float32)

                    h_med = float(np.median(seed_h_vals))
                    h_tol = float(max(8.0, min(15.0, np.std(seed_h_vals) * 1.8 + 6.0)))
                    s_low = float(max(20.0, np.percentile(seed_s_vals, 20) - 12.0))
                    v_low = float(max(8.0, np.percentile(seed_v_vals, 10) - 20.0))
                    v_high = float(min(235.0, np.percentile(seed_v_vals, 92) + 25.0))

                    adaptive_color_mask = (
                        (np.abs(Hc.astype(np.float32) - h_med) <= h_tol)
                        & (Sc.astype(np.float32) >= s_low)
                        & (V.astype(np.float32) >= v_low)
                        & (V.astype(np.float32) <= v_high)
                    )

    color_mask = color_mask | adaptive_color_mask

    # Remove global color-only path to avoid flooding ROI with false positives.
    combined_mask = (combined_strict | combined_soft_color_motion)
    combined_mask = combined_mask.astype(np.uint8) * 255

    roi_mask = make_blonding_roi_mask(
        (H, W),
        analysis_rect=analysis_rect,
        portafilter_ellipse=portafilter_ellipse,
    )
    if width_roi_mask is not None:
        if roi_mask is None:
            roi_mask = width_roi_mask
        else:
            roi_mask = cv2.bitwise_or(roi_mask, width_roi_mask)
    if roi_mask is not None:
        combined_mask = cv2.bitwise_and(combined_mask, roi_mask)

    exclusion_height = int(H * exclude_bottom_fraction)
    if exclusion_height > 0:
        combined_mask[-exclusion_height:, :] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_mask,
        connectivity=8
    )

    final_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    effective_min_area = max(8, int(min_component_area * 0.3))
    effective_motion_thresh = max(0.08, float(motion_fraction_thresh) * 0.45)
    accepted_components = []

    for label in range(1, num_labels):  # skip background

        component_mask = (labels == label)
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area < effective_min_area:
            continue

        motion_in_component = motion_mask_soft[component_mask]
        motion_fraction = np.sum(motion_in_component) / area
        color_in_component = color_mask[component_mask]
        color_fraction = np.sum(color_in_component) / area

        width_roi_fraction = 0.0
        if width_roi_mask is not None:
            width_in_component = (width_roi_mask[component_mask] > 0)
            width_roi_fraction = np.sum(width_in_component) / area

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]
        bbox_area = max(1, w * h)
        aspect_ratio = float(h) / float(max(1, w))
        fill_ratio = float(area) / float(bbox_area)
        comp_center_x = float(x + w / 2.0)
        center_offset_norm = abs(comp_center_x - width_roi_center_x) / max(1.0, width_roi_width / 2.0)

        seed_overlap = 0.0
        if seed_mask is not None:
            seed_overlap = float(np.sum((seed_mask > 0) & component_mask)) / float(max(1, area))

        origin_contact = False
        if width_roi_mask is not None and seed_mask is not None and seed_y_max is not None:
            origin_contact = seed_overlap > 0.0 or (y <= (seed_y_max + 3))
        
        if (
            motion_fraction < effective_motion_thresh
            and color_fraction < 0.24
            and width_roi_fraction < 0.55
        ):
            continue

        if width_roi_mask is not None:
            if width_roi_fraction < 0.45:
                continue
            if center_offset_norm > 0.95:
                continue
            if area > max(120, int(width_roi_area * 0.42)) and fill_ratio > 0.42:
                continue
            if aspect_ratio < 0.9 and not origin_contact:
                continue

        score = (
            (2.5 if origin_contact else 0.0)
            + min(aspect_ratio, 4.0) * 0.9
            + min(width_roi_fraction, 1.0) * 1.1
            + min(motion_fraction, 1.0) * 0.7
            + min(color_fraction, 1.0) * 0.6
            - center_offset_norm * 0.8
            - fill_ratio * 0.8
        )

        accepted_components.append((score, component_mask))

    if accepted_components:
        accepted_components.sort(key=lambda item: item[0], reverse=True)
        for _, component_mask in accepted_components[:2]:
            final_mask[component_mask] = 255
    else:
        # Conservative fallback: motion-dark regions only.
        final_mask = combined_strict.astype(np.uint8) * 255

    # Join fragmented thin stream sections along vertical direction.
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, vertical_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, open_kernel)

    if roi_mask is not None:
        final_mask = cv2.bitwise_and(final_mask, roi_mask)

    return final_mask


def draw_analysis_roi_on_frame(frame, analysis_rect, portafilter_ellipse=None, include_triangle=False):
    if analysis_rect is None and portafilter_ellipse is None:
        return frame

    label_x = 10
    label_y = 28
    if include_triangle:
        points = get_bottom_triangle_points_from_roi(analysis_rect, frame.shape[:2], keep_fraction=2.0 / 3.0)
        if points is not None:
            cv2.polylines(frame, [points], isClosed=True, color=(0, 255, 255), thickness=2)
            label_x = int(points[2][0]) - 15
            label_y = max(18, int(points[2][1]) - 8)

    if portafilter_ellipse is not None:
        try:
            cv2.ellipse(frame, portafilter_ellipse, (0, 255, 255), 2)
            label_x = int(portafilter_ellipse[0][0]) - 20
            label_y = max(18, int(portafilter_ellipse[0][1]) - 12)
        except Exception:
            pass

    cv2.putText(
        frame,
        "ROI",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def extract_colour_and_blonding(frames_bgr,
                                frames_gray,
                                start_frame,
                                end_frame,
                                fps=1.0,
                                mask_history=None,
                                detect_channeling=True,
                                portafilter_ellipse=None,
                                target_hole_size=None,
                                fast_params=None,
                                show_gui=True,
                                capture_frames=False,
                                analysis_roi=None):
    # Track stream colour statistics frame-by-frame and derive blonding timing/rate.
    # Reuse FAST settings from portafilter detection so both stages stay in sync.
    if fast_params is None:
        fast_params = {'threshold': 15, 'min_circularity': 0.4, 'size_tolerance': 5}

    if end_frame <= start_frame:
        raise ValueError("Invalid flow window")

    brightness_curve = []
    saturation_curve = []
    hue_curve = []
    channeling_counts = []
    channeling_frames = []
    stream_mask_frames = []
    stream_width_curve = []
    stream_straightness_curve = []
    stream_center_offset_curve = []
    stream_width_samples = []

    if analysis_roi is None and portafilter_ellipse is not None:
        analysis_roi = get_analysis_roi_from_ellipse(
            portafilter_ellipse,
            frames_bgr[0].shape,
            height_multiplier=2.0,
        )

    if analysis_roi is None:
        frame_h, frame_w = frames_bgr[0].shape[:2]
        analysis_roi = (0, 0, frame_w, frame_h)

    frame_shape = frames_bgr[0].shape[:2]
    channel_roi_mask = make_channeling_roi_mask(
        frame_shape,
        portafilter_ellipse=portafilter_ellipse,
    )
    blond_roi_mask = make_blonding_roi_mask(
        frame_shape,
        analysis_rect=analysis_roi,
        portafilter_ellipse=portafilter_ellipse,
    )
    channel_roi_pixels = int(np.count_nonzero(channel_roi_mask)) if channel_roi_mask is not None else 0
    blond_roi_pixels = int(np.count_nonzero(blond_roi_mask)) if blond_roi_mask is not None else 0
    stream_width_roi = get_stream_width_roi_from_ellipse(
        portafilter_ellipse,
        frames_bgr[0].shape,
    )

    if show_gui:
        print(f"Analysis ROI: x1={analysis_roi[0]}, y1={analysis_roi[1]}, x2={analysis_roi[2]}, y2={analysis_roi[3]}")

    prev_gray = frames_gray[start_frame]
    # Preserve prior blonding behavior by default (accumulate over full shot).
    if mask_history is None:
        history_len = max(1, end_frame - start_frame + 1)
    else:
        history_len = max(1, int(mask_history))
    mask_history_buffer = deque(maxlen=history_len)
    for t in range(start_frame + 1, end_frame + 1):
        curr_bgr = frames_bgr[t]
        curr_gray = frames_gray[t]

        vis_frame = curr_bgr.copy()
        hole_count = 0
        hole_keypoints = []

        if detect_channeling:
            hole_count, hole_keypoints = detect_channeling_in_frame(
                curr_gray,
                portafilter_ellipse=portafilter_ellipse,
                target_hole_size=target_hole_size,
                fast_params=fast_params,
                analysis_rect=analysis_roi,
            )
            if hole_count > 0:
                cv2.drawKeypoints(
                    vis_frame,
                    hole_keypoints,
                    vis_frame,
                    color=(0, 0, 255),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )

        channeling_counts.append(hole_count)
        draw_analysis_roi_on_frame(
            vis_frame,
            analysis_roi,
            portafilter_ellipse=portafilter_ellipse,
        )

        stream_mask = compute_stream_mask(
            prev_gray,
            curr_gray,
            curr_bgr,
            analysis_rect=analysis_roi,
            portafilter_ellipse=portafilter_ellipse,
            stream_width_roi=stream_width_roi,
        )

        width_measurement = detect_stream_width_in_roi(
            curr_bgr,
            curr_gray,
            stream_mask,
            stream_width_roi,
        )
        if width_measurement is None:
            stream_width_curve.append(np.nan)
            stream_straightness_curve.append(np.nan)
            stream_center_offset_curve.append(np.nan)
        else:
            stream_width_curve.append(float(width_measurement["width_px"]))
            stream_straightness_curve.append(float(width_measurement["straightness_score"]))
            stream_center_offset_curve.append(float(width_measurement["center_offset_px"]))
            stream_width_samples.append(
                {
                    "frame_index": int(t),
                    "width_px": float(width_measurement["width_px"]),
                    "measurement_row_y": int(width_measurement["measurement_row_y"]),
                    "measurement_row_fraction": float(width_measurement["measurement_row_fraction"]),
                    "straightness_score": float(width_measurement["straightness_score"]),
                    "centeredness_score": float(width_measurement["centeredness_score"]),
                    "center_offset_px": float(width_measurement["center_offset_px"]),
                    "vertical_line_score": float(width_measurement["vertical_line_score"]),
                    "edge_strength": float(width_measurement["edge_strength"]),
                    "center_std": float(width_measurement["center_std"]),
                    "center_slope": float(width_measurement["center_slope"]),
                    "veer_direction": str(width_measurement.get("veer_direction", "center")),
                    "veer_score": float(width_measurement.get("veer_score", 0.0)),
                    "row_count": int(width_measurement["row_count"]),
                }
            )

        draw_stream_width_overlay(
            vis_frame,
            width_roi=stream_width_roi,
            width_measurement=width_measurement,
        )

        text = f"Holes: {hole_count}"
        text_font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1
        text_thickness = 2
        text_size, text_baseline = cv2.getTextSize(text, text_font, text_scale, text_thickness)
        frame_h, frame_w = vis_frame.shape[:2]
        x = max(12, frame_w - text_size[0] - 12)
        y = frame_h - 12 - text_baseline
        cv2.putText(vis_frame, text, (x, y), text_font, text_scale, (0, 0, 0), 4)
        cv2.putText(vis_frame, text, (x, y), text_font, text_scale, (0, 255, 0), text_thickness)

        if capture_frames:
            channeling_frames.append(vis_frame.copy())
        if show_gui:
            cv2.imshow("Channeling Detection", vis_frame)

        mask_history_buffer.append(stream_mask)

        accumulated_mask = np.zeros_like(stream_mask, dtype=np.uint8)
        for mask in mask_history_buffer:
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask)

        mask_vis = render_stream_detection_preview(
            curr_bgr,
            analysis_rect=analysis_roi,
            portafilter_ellipse=portafilter_ellipse,
            stream_width_roi=stream_width_roi,
            stream_mask=stream_mask,
            width_measurement=width_measurement,
        )

        if capture_frames:
            stream_mask_frames.append(mask_vis.copy())
        if show_gui:
            cv2.imshow("Stream Mask", mask_vis)
        hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)

        if np.sum(accumulated_mask) == 0:
            brightness_curve.append(np.nan)
            saturation_curve.append(np.nan)
            hue_curve.append(np.nan)
        else:
            stream_pixels = hsv[accumulated_mask > 0]

            hue_curve.append(np.mean(stream_pixels[:, 0]))
            saturation_curve.append(np.mean(stream_pixels[:, 1]))
            brightness_curve.append(np.mean(stream_pixels[:, 2]))

        prev_gray = curr_gray

    brightness_curve = np.array(brightness_curve)
    saturation_curve = np.array(saturation_curve)
    hue_curve = np.array(hue_curve)

    valid = ~np.isnan(brightness_curve)

    if np.sum(valid) < 3:
        return None

    norm_brightness = (
        brightness_curve - np.nanmin(brightness_curve)
    ) / (np.nanmax(brightness_curve) - np.nanmin(brightness_curve) + 1e-6)

    norm_brightness_temp = norm_brightness.copy()
    for i in range(len(norm_brightness_temp)):
        if np.isnan(norm_brightness_temp[i]):
            valid_indices = np.where(~np.isnan(norm_brightness))[0]
            if len(valid_indices) > 0:
                nearest = valid_indices[np.argmin(np.abs(valid_indices - i))]
                norm_brightness_temp[i] = norm_brightness[nearest]
            else:
                norm_brightness_temp[i] = 0.5
    
    norm_brightness = np.convolve(
        norm_brightness_temp,
        np.ones(3) / 3,
        mode='same'
    )

    # Blonding point is where brightness rises fastest after the initial pour transient.
    derivative = np.gradient(norm_brightness)
    derivative = np.convolve(derivative, np.ones(3) / 3, mode='same')

    sample_count = len(norm_brightness)
    warmup = min(max(3, int(sample_count * 0.15)), max(0, sample_count - 1))
    candidate_indices = np.arange(warmup, sample_count)
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(sample_count)

    # Ignore very dark early values to avoid false blonding picks during flow onset.
    brightness_gate = norm_brightness[candidate_indices] >= 0.25
    gated_indices = candidate_indices[brightness_gate]
    if len(gated_indices) > 0:
        candidate_indices = gated_indices

    blond_idx = int(candidate_indices[np.argmax(derivative[candidate_indices])])
    blond_rate = float(derivative[blond_idx])

    blond_frame = start_frame + 1 + blond_idx

    best_stream_width = None
    if stream_width_samples:
        best_stream_width = min(
            stream_width_samples,
            key=lambda row: (
                float(row["straightness_score"]),
                float(row["centeredness_score"]),
                float(row["center_std"]),
                -int(row["row_count"]),
            ),
        )

    # Keep this function non-blocking and GUI-safe by avoiding matplotlib output.
    # Plotting and saving files are now handled in the application layer.

    return {
        "brightness_curve": brightness_curve,
        "saturation_curve": saturation_curve,
        "hue_curve": hue_curve,
        "blond_score_curve": norm_brightness,
        "blond_frame": blond_frame,
        "blond_rate": float(blond_rate),
        "channeling_counts": channeling_counts,
        "channeling_frames": channeling_frames if capture_frames else None,
        "stream_mask_frames": stream_mask_frames if capture_frames else None,
        "stream_width_curve": np.array(stream_width_curve, dtype=float),
        "stream_straightness_curve": np.array(stream_straightness_curve, dtype=float),
        "stream_center_offset_curve": np.array(stream_center_offset_curve, dtype=float),
        "stream_centeredness_curve": np.abs(np.array(stream_center_offset_curve, dtype=float)),
        "stream_width_at_straightest_px": float(best_stream_width["width_px"]) if best_stream_width else None,
        "stream_width_straightest_frame": int(best_stream_width["frame_index"]) if best_stream_width else None,
        "stream_width_center_offset_px": float(best_stream_width["center_offset_px"]) if best_stream_width else None,
        "stream_width_centeredness_score": float(best_stream_width["centeredness_score"]) if best_stream_width else None,
        "stream_width_straightness_score": float(best_stream_width["straightness_score"]) if best_stream_width else None,
        "stream_veer_direction": str(best_stream_width["veer_direction"]) if best_stream_width else None,
        "stream_veer_score": float(best_stream_width["veer_score"]) if best_stream_width else None,
        "stream_width_sample_rows": int(best_stream_width["row_count"]) if best_stream_width else None,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "fps": fps,
        "analysis_roi": {
            "x1": int(analysis_roi[0]),
            "y1": int(analysis_roi[1]),
            "x2": int(analysis_roi[2]),
            "y2": int(analysis_roi[3]),
        },
        "channel_roi_pixels": channel_roi_pixels,
        "blond_roi_pixels": blond_roi_pixels,
        "stream_width_roi": {
            "x1": int(stream_width_roi[0]),
            "y1": int(stream_width_roi[1]),
            "x2": int(stream_width_roi[2]),
            "y2": int(stream_width_roi[3]),
        } if stream_width_roi is not None else None,
        "portafilter_ellipse": {
            "cx": float(portafilter_ellipse[0][0]),
            "cy": float(portafilter_ellipse[0][1]),
            "width": float(portafilter_ellipse[1][0]),
            "height": float(portafilter_ellipse[1][1]),
            "angle": float(portafilter_ellipse[2]),
        } if portafilter_ellipse is not None else None,
    }


def extract_features_from_video(cropped_frames_dir=None, 
                               video_name="analysis",
                               output_dir=None,
                               output_blond_dir=None,
                               output_channeling_dir=None,
                               output_results_dir=None,
                               save_plots=True,
                               detect_channeling=True,
                               show_gui=True,
                               capture_channeling_frames=True,
                               portafilter_override_ellipse=None,
                               portafilter_override_hole_size=None,
                               portafilter_override_fast_params=None):
    # Extract blonding and channeling features from cropped frames.
    if cropped_frames_dir is None:
        cropped_frames_dir = Crop_Dir
    if output_dir is None:
        output_dir = Output_Dir
    if output_results_dir is None:
        output_results_dir = output_dir

    print(f"Loading frames")
    frames_bgr, frames_gray, frame_names = load_frames(cropped_frames_dir)
    print(f"Loaded {len(frames_gray)} frames")

    H, W = frames_gray[0].shape
    roi = (0, H, 0, W)

    print("Detecting flow start/end")
    start_frame, end_frame = detect_flow_start_end(
        frames_gray=frames_gray,
        roi=roi,
        pixel_diff_thresh=15,           # Lower threshold for subtle changes
        change_fraction_thresh=0.005,   # Lower threshold for slow flow
        start_consec_frames=2,          # Still need quick response for start
        end_consec_frames=5,            # More frames to confirm end (handles choked shots)
        smoothing_window=3,             # Smooth out noise
        end_lookback_window=10          # Rolling average for robust end detection
    )

    print("\nFlow Detection Results:")
    print(f"Flow start frame index : {start_frame}")
    print(f"Flow end frame index   : {end_frame}")
    print(f"Shot duration (frames) : {end_frame - start_frame + 1}")
    print(f"Start frame file       : {frame_names[start_frame]}")
    print(f"End frame file         : {frame_names[end_frame]}")

    # Detect portafilter for the cropped ROI context
    fast_params = portafilter_override_fast_params
    if portafilter_override_ellipse is not None:
        print("Using overridden portafilter ellipse for channeling detection.")
        portafilter_ellipse = portafilter_override_ellipse
        hole_mode_size = portafilter_override_hole_size
    else:
        print("Detecting portafilter ellipse for channeling detection...")
        pf_second_idx = min(start_frame + 20, len(frames_bgr) - 1)
        pf_second_frame = frames_bgr[pf_second_idx] if pf_second_idx > start_frame else None

        _, portafilter_ellipse, hole_mode_size, fast_params = detect_elliptical_portafilter_with_holes(
            frames_bgr[start_frame],
            save_dashboard=False,
            use_interactive=False,
            second_frame=pf_second_frame,
            mask_threshold=15,
        )

    if fast_params is None:
        fast_params = {'threshold': 15, 'min_circularity': 0.4, 'size_tolerance': 5}

    print("Extracting color and blonding features")

    colour_results = extract_colour_and_blonding(
        frames_bgr,
        frames_gray,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=1.0,
        detect_channeling=detect_channeling,
        portafilter_ellipse=portafilter_ellipse,
        target_hole_size=hole_mode_size,
        fast_params=fast_params,
        show_gui=show_gui,
        capture_frames=capture_channeling_frames
    )

    print("\nBlonding And Channeling Results:")
    if colour_results:
        def to_json_curve(values):
            # Convert numpy-heavy arrays to plain JSON-safe lists.
            if values is None:
                return []
            serialised = []
            for value in values:
                try:
                    number = float(value)
                    if np.isnan(number):
                        serialised.append(None)
                    else:
                        serialised.append(number)
                except Exception:
                    serialised.append(None)
            return serialised

        channeling_counts = to_json_curve(colour_results.get("channeling_counts"))
        valid_channeling = [v for v in channeling_counts if v is not None]
        stream_width_curve = to_json_curve(colour_results.get("stream_width_curve"))
        valid_stream_widths = [v for v in stream_width_curve if v is not None]
        stream_center_offset_curve = to_json_curve(colour_results.get("stream_center_offset_curve"))
        valid_stream_offsets = [v for v in stream_center_offset_curve if v is not None]

        for k, v in colour_results.items():
            if k == "channeling_counts":
                if valid_channeling:
                    channeling_array = np.array(valid_channeling, dtype=float)
                    print(f"Channeling stats:")
                    print(f"  - Average visible holes: {np.mean(channeling_array):.2f}")
                    print(f"  - Max visible holes: {np.max(channeling_array):.0f}")
                    print(f"  - Min visible holes: {np.min(channeling_array):.0f}")
                else:
                    print("Channeling stats: no valid hole detections")
            elif k == "stream_width_curve":
                if valid_stream_widths:
                    width_array = np.array(valid_stream_widths, dtype=float)
                    print("Stream width stats (px):")
                    print(f"  - Average width: {np.mean(width_array):.2f}")
                    print(f"  - Max width: {np.max(width_array):.2f}")
                    print(f"  - Min width: {np.min(width_array):.2f}")
                    if colour_results.get("stream_width_at_straightest_px") is not None:
                        print(
                            "  - Straightest-frame width: "
                            f"{float(colour_results['stream_width_at_straightest_px']):.2f} "
                            f"(frame {int(colour_results['stream_width_straightest_frame'])})"
                        )
                    if colour_results.get("stream_width_center_offset_px") is not None:
                        print(
                            "  - Straightest-frame center offset: "
                            f"{float(colour_results['stream_width_center_offset_px']):+.2f} px"
                        )
                else:
                    print("Stream width stats: no valid edge detections")
            elif k not in [
                "brightness_curve",
                "saturation_curve",
                "hue_curve",
                "blond_score_curve",
                "channeling_frames",
                "stream_mask_frames",
                "stream_straightness_curve",
                "stream_center_offset_curve",
                "stream_centeredness_curve",
            ]:
                print(f"{k}: {v}")
        
        # Save results as JSON
        results_json = {
            "video_name": video_name,
            "blond_frame": int(colour_results["blond_frame"]) if colour_results.get("blond_frame") is not None else None,
            "blond_rate": float(colour_results["blond_rate"]) if colour_results.get("blond_rate") is not None else None,
            "flow_start": int(start_frame),
            "flow_end": int(end_frame),
            "total_frames": len(frames_gray),
            "fps": float(colour_results.get("fps", 1.0) or 1.0),
            "analysis_roi": colour_results.get("analysis_roi"),
            "channel_roi_pixels": int(colour_results.get("channel_roi_pixels") or 0),
            "blond_roi_pixels": int(colour_results.get("blond_roi_pixels") or 0),
            "stream_width_roi": colour_results.get("stream_width_roi"),
            "portafilter_ellipse": colour_results.get("portafilter_ellipse"),
            "target_hole_size": float(hole_mode_size) if hole_mode_size is not None else None,
            "brightness_curve": to_json_curve(colour_results.get("brightness_curve")),
            "saturation_curve": to_json_curve(colour_results.get("saturation_curve")),
            "hue_curve": to_json_curve(colour_results.get("hue_curve")),
            "blond_score_curve": to_json_curve(colour_results.get("blond_score_curve")),
            "channeling_counts": channeling_counts,
            "stream_width_curve": stream_width_curve,
            "stream_straightness_curve": to_json_curve(colour_results.get("stream_straightness_curve")),
            "stream_center_offset_curve": stream_center_offset_curve,
            "stream_centeredness_curve": to_json_curve(colour_results.get("stream_centeredness_curve")),
            "stream_width_at_straightest_px": (
                float(colour_results["stream_width_at_straightest_px"])
                if colour_results.get("stream_width_at_straightest_px") is not None
                else None
            ),
            "stream_width_straightest_frame": (
                int(colour_results["stream_width_straightest_frame"])
                if colour_results.get("stream_width_straightest_frame") is not None
                else None
            ),
            "stream_width_straightness_score": (
                float(colour_results["stream_width_straightness_score"])
                if colour_results.get("stream_width_straightness_score") is not None
                else None
            ),
            "stream_veer_direction": colour_results.get("stream_veer_direction"),
            "stream_veer_score": (
                float(colour_results["stream_veer_score"])
                if colour_results.get("stream_veer_score") is not None
                else None
            ),
            "stream_width_center_offset_px": (
                float(colour_results["stream_width_center_offset_px"])
                if colour_results.get("stream_width_center_offset_px") is not None
                else None
            ),
            "stream_width_centeredness_score": (
                float(colour_results["stream_width_centeredness_score"])
                if colour_results.get("stream_width_centeredness_score") is not None
                else None
            ),
            "stream_width_sample_rows": (
                int(colour_results["stream_width_sample_rows"])
                if colour_results.get("stream_width_sample_rows") is not None
                else None
            ),
        }
        if valid_channeling:
            channeling_array = np.array(valid_channeling, dtype=float)
            results_json["channeling_stats"] = {
                "average": float(np.mean(channeling_array)),
                "max": int(np.max(channeling_array)),
                "min": int(np.min(channeling_array))
            }
        if valid_stream_widths:
            stream_width_array = np.array(valid_stream_widths, dtype=float)
            results_json["stream_width_stats"] = {
                "average": float(np.mean(stream_width_array)),
                "max": float(np.max(stream_width_array)),
                "min": float(np.min(stream_width_array)),
            }
        if valid_stream_offsets:
            stream_offset_array = np.array(valid_stream_offsets, dtype=float)
            results_json["stream_center_offset_stats"] = {
                "average": float(np.mean(stream_offset_array)),
                "max_right": float(np.max(stream_offset_array)),
                "max_left": float(np.min(stream_offset_array)),
            }
        
        os.makedirs(output_results_dir, exist_ok=True)
        json_output_path = os.path.join(output_results_dir, f"{video_name}_results.json")
        with open(json_output_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nResults saved to {json_output_path}")

    if show_gui:
        cv2.destroyAllWindows()
    
    return colour_results


if __name__ == "__main__":
    import sys
    
    # Allow command-line usage
    video_name = "test2"
    cropped_dir =  None
    
    print(f"Extracting features for video: {video_name}")
    results = extract_features_from_video(cropped_dir, video_name)
