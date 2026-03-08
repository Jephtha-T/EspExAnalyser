import os
import cv2
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from Portafilter_Detection import detect_elliptical_portafilter_with_holes

# Helpers

base_dir = os.path.dirname(os.path.abspath(__file__))
crop_dir = os.path.join(base_dir, "Image Data", "Cropped")
out_dir = os.path.join(base_dir, "Analysis")
os.makedirs(out_dir, exist_ok=True)

def moving_average(x, w=5):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='same')

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
    y1, y2, x1, x2 = roi
    roi_area = (y2 - y1) * (x2 - x1)

    if roi_area <= 0:
        raise ValueError("Invalid ROI dimensions")

    change_fractions = []
    change_masks = []

    # Calculate frame-to-frame changes
    for i in range(1, len(frames_gray)):
        prev = frames_gray[i - 1][y1:y2, x1:x2].astype(np.int16)
        curr = frames_gray[i][y1:y2, x1:x2].astype(np.int16)

        diff = np.abs(curr - prev)
        mask = (diff > pixel_diff_thresh).astype(np.uint8) * 255

        change_fraction = np.sum(mask > 0) / roi_area

        change_fractions.append(change_fraction)
        change_masks.append(mask)

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


def generate_color_change_mask(frame1_bgr, frame2_bgr, threshold=30):
    # Mask pixels that changed color between two frames
    if frame1_bgr.shape != frame2_bgr.shape:
        frame1_bgr = cv2.resize(frame1_bgr, (frame2_bgr.shape[1], frame2_bgr.shape[0]))

    lab1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2LAB)

    diff = cv2.absdiff(lab1, lab2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    return mask


def create_espresso_zone_mask(frames_bgr, start_frame, reference_frame_offset=20, 
                               color_threshold=30, exclude_bottom_fraction=0.15):
    # Build a rough mask of where espresso shows up
    ref_early = frames_bgr[start_frame]
    ref_late_idx = min(start_frame + reference_frame_offset, len(frames_bgr) - 1)
    ref_late = frames_bgr[ref_late_idx]
    
    espresso_zone = generate_color_change_mask(ref_early, ref_late, threshold=color_threshold)
    
    H, W = espresso_zone.shape
    exclusion_height = int(H * exclude_bottom_fraction)
    if exclusion_height > 0:
        espresso_zone[-exclusion_height:, :] = 0
    
    return espresso_zone


def detect_fast_holes(image, threshold=25, min_circularity=0.4):
    # Find circular FAST keypoints (holes)
    try:
        fast_create = getattr(cv2, 'FastFeatureDetector_create', None)
        if fast_create is not None:
            fast = fast_create(threshold=threshold)
        else:
            fast_class = getattr(cv2, 'FastFeatureDetector', None)
            if fast_class is not None:
                fast = fast_class.create(threshold=threshold)
            else:
                raise RuntimeError('No FAST detector available in this OpenCV build')
        
        keypoints = fast.detect(image, None)
    except Exception as e:
        print(f"FAST detection failed: {e}")
        return [], []
    
    if keypoints:
        filtered_keypoints = []
        keypoint_sizes = []
        height, width = image.shape[:2]
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            if hasattr(kp, 'size'):
                size = int(kp.size)
            else:
                size = 20
            
            if x < size//2 or y < size//2 or x >= width - size//2 or y >= height - size//2:
                continue
            
            x1, y1 = max(0, int(x - size//2)), max(0, int(y - size//2))
            x2, y2 = min(width, int(x + size//2)), min(height, int(y + size//2))
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            try:
                _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > 0:
                        perimeter = cv2.arcLength(largest_contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            if circularity >= min_circularity:
                                filtered_keypoints.append(kp)
                                keypoint_sizes.append(size)
            except Exception:
                pass
        
        return filtered_keypoints, keypoint_sizes
    
    return [], []


def find_mode_size(keypoint_sizes, tolerance=5):
    # Get the most common keypoint size
    if not keypoint_sizes:
        return None
    
    # Group sizes within tolerance
    size_groups = {}
    for size in keypoint_sizes:
        grouped = False
        for group_size in size_groups:
            if abs(size - group_size) <= tolerance:
                size_groups[group_size] += 1
                grouped = True
                break
        if not grouped:
            size_groups[size] = 1
    
    # Find the most common size
    if size_groups:
        mode_size = max(size_groups.keys(), key=lambda k: size_groups[k])
        return mode_size
    
    return None


def filter_keypoints_by_size(keypoints, keypoint_sizes, target_size, tolerance=5):
    # Keep only keypoints near the target size
    if target_size is None:
        return keypoints
    
    filtered_keypoints = []
    for kp, size in zip(keypoints, keypoint_sizes):
        if abs(size - target_size) <= tolerance:
            filtered_keypoints.append(kp)
    
    return filtered_keypoints


def detect_channeling_in_frame(frame_gray, portafilter_ellipse=None, 
                                target_hole_size=None, fast_params=None):
    # Count visible holes inside the portafilter ellipse region.
    # Use FAST parameters from portafilter detection to ensure consistency
    if fast_params is None:
        fast_params = {'threshold': 15, 'min_circularity': 0.4, 'size_tolerance': 5}
    
    mask = None
    if portafilter_ellipse is not None:
        H, W = frame_gray.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, portafilter_ellipse, 255, -1)
        
        # Apply mask to image - zeros out everything outside the ellipse
        masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    else:
        masked_frame = frame_gray
    
    # Detect circular features (holes) using exact FAST params from portafilter detection
    keypoints, keypoint_sizes = detect_fast_holes(
        masked_frame, 
        threshold=fast_params['threshold'],
        min_circularity=fast_params['min_circularity']
    )
    
    # Filter keypoints to only those inside the ellipse mask (spatial filtering)
    if portafilter_ellipse is not None and mask is not None and len(keypoints) > 0:
        original_count = len(keypoints)
        filtered_by_roi = []
        filtered_sizes_by_roi = []
        
        for kp, size in zip(keypoints, keypoint_sizes):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            # Check if keypoint is within image bounds and inside the ellipse
            if 0 <= y < H and 0 <= x < W and mask[y, x] == 255:
                filtered_by_roi.append(kp)
                filtered_sizes_by_roi.append(size)
        
        keypoints = filtered_by_roi
        keypoint_sizes = filtered_sizes_by_roi
        
        # Debug: show how many keypoints were filtered out by ROI
        if original_count > len(keypoints):
            filtered_out = original_count - len(keypoints)
            # Only print occasionally to avoid spam
            # print(f"Filtered out {filtered_out} keypoints outside ellipse ROI")
    
    # Filter to target hole size using the same tolerance from portafilter detection
    if target_hole_size is not None and len(keypoints) > 0:
        filtered_keypoints = filter_keypoints_by_size(
            keypoints, 
            keypoint_sizes, 
            target_hole_size, 
            tolerance=fast_params['size_tolerance']
        )
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
                        exclude_bottom_fraction=0.15,
                        espresso_zone_mask=None):
    # Build a mask for the stream using motion and color

    H, W = curr_gray.shape

    diff = cv2.absdiff(curr_gray, prev_gray)
    motion_mask = diff > diff_thresh

    hsv = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[:, :, 2]
    dark_mask = V < brightness_thresh
    dark_mask = dark_mask.astype(np.uint8)

    combined_mask = (dark_mask > 0) & motion_mask
    combined_mask = combined_mask.astype(np.uint8) * 255

    if espresso_zone_mask is not None:
        combined_mask = cv2.bitwise_or(combined_mask, espresso_zone_mask)

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

    return final_mask


def extract_colour_and_blonding(frames_bgr,
                                frames_gray,
                                start_frame,
                                end_frame,
                                fps=1.0,
                                plot=True,
                                plot_output_blond_path=None,
                                plot_output_channeling_path=None,
                                mask_history=2,
                                use_color_zone_mask=True,
                                zone_reference_offset=20,
                                zone_color_threshold=30,
                                detect_channeling=True,
                                portafilter_ellipse=None,
                                target_hole_size=None,
                                fast_params=None,
                                show_gui=True):
    # Track stream color and optional channeling over time
    # Extract FAST parameters from portafilter detection (use defaults if not provided)
    if fast_params is None:
        fast_params = {'threshold': 15, 'min_circularity': 0.4, 'size_tolerance': 5}

    if end_frame <= start_frame:
        raise ValueError("Invalid flow window")

    brightness_curve = []
    saturation_curve = []
    hue_curve = []
    channeling_counts = []
    espresso_zone_mask = None
    if use_color_zone_mask:
        print("Creating espresso zone mask from color change...")
        espresso_zone_mask = create_espresso_zone_mask(
            frames_bgr,
            start_frame,
            reference_frame_offset=zone_reference_offset,
            color_threshold=zone_color_threshold,
            exclude_bottom_fraction=0.15
        )
        print(f"Espresso zone covers {np.sum(espresso_zone_mask > 0)} pixels")
        
        if show_gui:
            cv2.imshow("Espresso Zone Mask", espresso_zone_mask)

    prev_gray = frames_gray[start_frame]
    mask_history_buffer = []
    for t in range(start_frame + 1, end_frame + 1):

        curr_bgr = frames_bgr[t]
        curr_gray = frames_gray[t]
        if detect_channeling:
            hole_count, hole_keypoints = detect_channeling_in_frame(
                curr_gray,
                portafilter_ellipse=portafilter_ellipse,
                target_hole_size=target_hole_size,
                fast_params=fast_params
            )
            channeling_counts.append(hole_count)
            
            if hole_count > 0:
                vis_frame = curr_bgr.copy()
                cv2.drawKeypoints(vis_frame, hole_keypoints, vis_frame, 
                                color=(0, 0, 255), 
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.putText(vis_frame, f"Holes: {hole_count}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 255), 2)
                if show_gui:
                    cv2.imshow("Channeling Detection", vis_frame)
        else:
            channeling_counts.append(0)

        stream_mask = compute_stream_mask(
            prev_gray,
            curr_gray,
            curr_bgr,
            espresso_zone_mask=espresso_zone_mask
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

    derivative = np.gradient(norm_brightness)
    blond_idx = np.nanargmax(derivative)
    blond_rate = derivative[blond_idx]

    blond_frame = start_frame + 1 + blond_idx

    if plot:
        time_axis = np.arange(len(norm_brightness)) / fps

        if detect_channeling and len(channeling_counts) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            ax1.plot(time_axis, norm_brightness, label="Brightness (norm)")
            ax1.plot(time_axis, saturation_curve / 255.0, label="Saturation (norm)")
            ax1.axvline(
                x=(blond_frame - start_frame) / fps,
                color='r',
                linestyle='--',
                label="Blonding point"
            )
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Normalised Value")
            ax1.set_title("Espresso Stream Colour Transition")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            channeling_array = np.array(channeling_counts)
            ax2.plot(time_axis, channeling_array, color='red', linewidth=2, label="Visible Holes")
            ax2.fill_between(time_axis, channeling_array, alpha=0.3, color='red')
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Hole Count")
            ax2.set_title("Channeling Detection (Visible Portafilter Holes)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            avg_holes = np.mean(channeling_array)
            max_holes = np.max(channeling_array)
            ax2.text(0.02, 0.98, f"Avg holes: {avg_holes:.1f}\nMax holes: {max_holes}", 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            if plot_output_blond_path:
                plt.savefig(plot_output_blond_path, dpi=150, bbox_inches='tight')
                print(f"Blonding plot saved to {plot_output_blond_path}")
            if plot_output_channeling_path:
                # Save just the channeling subplot
                fig2, ax2_save = plt.subplots(figsize=(12, 5))
                ax2_save.plot(time_axis, channeling_array, color='red', linewidth=2, label="Visible Holes")
                ax2_save.fill_between(time_axis, channeling_array, alpha=0.3, color='red')
                ax2_save.set_xlabel("Time (s)")
                ax2_save.set_ylabel("Hole Count")
                ax2_save.set_title("Channeling Detection (Visible Portafilter Holes)")
                ax2_save.legend()
                ax2_save.grid(True, alpha=0.3)
                ax2_save.text(0.02, 0.98, f"Avg holes: {avg_holes:.1f}\nMax holes: {max_holes}", 
                            transform=ax2_save.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                plt.tight_layout()
                fig2.savefig(plot_output_channeling_path, dpi=150, bbox_inches='tight')
                print(f"Channeling plot saved to {plot_output_channeling_path}")
                plt.close(fig2)
            if show_gui:
                plt.show()
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(10, 5))
            plt.plot(time_axis, norm_brightness, label="Brightness (norm)")
            plt.plot(time_axis, saturation_curve / 255.0, label="Saturation (norm)")
            plt.axvline(
                x=(blond_frame - start_frame) / fps,
                color='r',
                linestyle='--',
                label="Blonding point"
            )
            plt.xlabel("Time (s)")
            plt.ylabel("Normalised Value")
            plt.title("Espresso Stream Colour Transition")
            plt.legend()
            plt.tight_layout()
            if plot_output_blond_path:
                fig.savefig(plot_output_blond_path, dpi=150, bbox_inches='tight')
                print(f"Blonding plot saved to {plot_output_blond_path}")
            if show_gui:
                plt.show()
            plt.close(fig)

    return {
        "brightness_curve": brightness_curve,
        "saturation_curve": saturation_curve,
        "hue_curve": hue_curve,
        "blond_frame": blond_frame,
        "blond_rate": float(blond_rate),
        "channeling_counts": channeling_counts
    }


def extract_features_from_video(cropped_frames_dir=None, 
                               video_name="analysis",
                               output_dir=None,
                               output_blond_dir=None,
                               output_channeling_dir=None,
                               output_results_dir=None,
                               save_plots=True,
                               detect_channeling=True,
                               show_gui=True):
    # Extract blonding and channeling features from cropped frames.
    if cropped_frames_dir is None:
        cropped_frames_dir = crop_dir
    if output_dir is None:
        output_dir = out_dir
    if output_blond_dir is None:
        output_blond_dir = output_dir
    if output_channeling_dir is None:
        output_channeling_dir = output_dir
    if output_results_dir is None:
        output_results_dir = output_dir

    print(f"Loading frames from {cropped_frames_dir}")
    frames_bgr, frames_gray, frame_names = load_frames(cropped_frames_dir)
    print(f"Loaded {len(frames_gray)} frames")

    H, W = frames_gray[0].shape
    roi = (0, H, 0, W)

    print("Detecting flow start/end...")
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

    print("\n===== FLOW DETECTION RESULT =====")
    print(f"Flow start frame index : {start_frame}")
    print(f"Flow end frame index   : {end_frame}")
    print(f"Shot duration (frames) : {end_frame - start_frame + 1}")
    print(f"Start frame file       : {frame_names[start_frame]}")
    print(f"End frame file         : {frame_names[end_frame]}")

    # Detect portafilter for the cropped ROI context
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
    
    print(f"Using FAST detection params from portafilter: threshold={fast_params['threshold']}, "
          f"circularity={fast_params['min_circularity']}, tolerance={fast_params['size_tolerance']}")

    print("Extracting color and blonding features...")
    
    # Prepare output paths
    plot_output_blond_path = None
    plot_output_channeling_path = None
    if save_plots:
        os.makedirs(output_blond_dir, exist_ok=True)
        plot_output_blond_path = os.path.join(output_blond_dir, f"{video_name}_Blond.png")

        if detect_channeling:
            os.makedirs(output_channeling_dir, exist_ok=True)
            plot_output_channeling_path = os.path.join(output_channeling_dir, f"{video_name}_Channeling.png")
    
    colour_results = extract_colour_and_blonding(
        frames_bgr,
        frames_gray,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=1.0,
        plot=True,
        plot_output_blond_path=plot_output_blond_path,
        plot_output_channeling_path=plot_output_channeling_path,
        detect_channeling=detect_channeling,
        portafilter_ellipse=portafilter_ellipse,
        target_hole_size=hole_mode_size,
        fast_params=fast_params,
        show_gui=show_gui
    )

    print("\n===== BLONDING AND CHANNELING FEATURES =====")
    if colour_results:
        for k, v in colour_results.items():
            if k == "channeling_counts":
                channeling_array = np.array(v)
                print(f"Channeling stats:")
                print(f"  - Average visible holes: {np.mean(channeling_array):.2f}")
                print(f"  - Max visible holes: {np.max(channeling_array)}")
                print(f"  - Min visible holes: {np.min(channeling_array)}")
            elif k not in ["brightness_curve", "saturation_curve", "hue_curve"]:
                print(f"{k}: {v}")
        
        # Save results as JSON
        results_json = {
            "video_name": video_name,
            "blond_frame": int(colour_results["blond_frame"]) if colour_results.get("blond_frame") is not None else None,
            "blond_rate": float(colour_results["blond_rate"]) if colour_results.get("blond_rate") is not None else None,
            "flow_start": int(start_frame),
            "flow_end": int(end_frame),
            "total_frames": len(frames_gray)
        }
        if "channeling_counts" in colour_results:
            channeling_array = np.array(colour_results["channeling_counts"])
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
