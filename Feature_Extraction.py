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
                        portafilter_ellipse=None):
    # Stream mask = moving dark pixels, then clipped to blonding ROI union.

    H, W = curr_gray.shape

    diff = cv2.absdiff(curr_gray, prev_gray)
    motion_mask = diff > diff_thresh

    hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    dark_mask = V < brightness_thresh
    dark_mask = dark_mask.astype(np.uint8)

    combined_mask = (dark_mask > 0) & motion_mask
    combined_mask = combined_mask.astype(np.uint8) * 255

    exclusion_height = int(H * exclude_bottom_fraction)
    if exclusion_height > 0:
        combined_mask[-exclusion_height:, :] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_mask,
        connectivity=8
    )

    final_mask = np.zeros_like(combined_mask, dtype=np.uint8)

    for label in range(1, num_labels):  # skip background

        component_mask = (labels == label)
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area < min_component_area:
            continue

        motion_in_component = motion_mask[component_mask]
        motion_fraction = np.sum(motion_in_component) / area
        
        if motion_fraction < motion_fraction_thresh:
            continue

        # This component passed - it's flowing espresso
        final_mask[component_mask] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    roi_mask = make_blonding_roi_mask(
        (H, W),
        analysis_rect=analysis_rect,
        portafilter_ellipse=portafilter_ellipse,
    )
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

        stream_mask = compute_stream_mask(
            prev_gray,
            curr_gray,
            curr_bgr,
            analysis_rect=analysis_roi,
            portafilter_ellipse=portafilter_ellipse,
        )

        mask_history_buffer.append(stream_mask)

        accumulated_mask = np.zeros_like(stream_mask, dtype=np.uint8)
        for mask in mask_history_buffer:
            accumulated_mask = cv2.bitwise_or(accumulated_mask, mask)

        if show_gui:
            cv2.imshow("Stream Mask", accumulated_mask)
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
            elif k not in ["brightness_curve", "saturation_curve", "hue_curve", "blond_score_curve", "channeling_frames"]:
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
            "portafilter_ellipse": colour_results.get("portafilter_ellipse"),
            "target_hole_size": float(hole_mode_size) if hole_mode_size is not None else None,
            "brightness_curve": to_json_curve(colour_results.get("brightness_curve")),
            "saturation_curve": to_json_curve(colour_results.get("saturation_curve")),
            "hue_curve": to_json_curve(colour_results.get("hue_curve")),
            "blond_score_curve": to_json_curve(colour_results.get("blond_score_curve")),
            "channeling_counts": channeling_counts,
        }
        if valid_channeling:
            channeling_array = np.array(valid_channeling, dtype=float)
            results_json["channeling_stats"] = {
                "average": float(np.mean(channeling_array)),
                "max": int(np.max(channeling_array)),
                "min": int(np.min(channeling_array))
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
