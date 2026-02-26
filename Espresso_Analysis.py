import os
import cv2
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STAB_DIR = os.path.join(BASE_DIR, "Image Data", "Stabilised")
OUT_DIR = os.path.join(BASE_DIR, "Image Data", "Analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# Parameters (tune these to your camera / shots)
FPS = 30.0                 # approximate fps of the original video (used only if you want time scaling)
FRAME_STEP = 1             # if frames are every second, keep 1; else adjust accordingly
TOP_ROI_FRACTION = 0.66    # proportion of cropped image height used as basket ROI (top area)
DIFF_FLOW_THRESH = 6.0     # mean-abs-diff threshold that indicates flow present
DIFF_FLOW_MIN_CONSEC = 2   # number of consecutive frames over threshold to count as flow start/end
BLUR_KERNEL = (5,5)
BRIGHT_SMOOTH_WINDOW = 5   # moving average window for brightness curve
STREAM_WIDTH_SMOOTH = 3
MIN_STREAM_AREA = 20       # small area filter for blobs (px)
DRY_SPOT_PERSISTANCE = 0.25 # fraction of frames a "bright pixel" stays bright -> treated as dry-spot
DRY_SPOT_THRESH = 200      # intensity threshold for 'dry' (on grayscale) 0-255
MULTI_BLOB_FRAC_THRESH = 0.2 # fraction of frames with >1 blobs to raise spraying anomaly

# Classification thresholds (heuristic, tune for your dataset)
MIN_IDEAL_TIME = 18.0      # seconds -> below this tends toward UNDER
MAX_IDEAL_TIME = 24.0      # seconds -> above this tends toward OVER
FAST_BLOND_RATE = 0.5      # change in normalized brightness per second considered "fast blonding"
SLOW_BLOND_RATE = 0.05     # extremely slow blonding (could be stuck/blocked) - not used directly here

# ---------- Helpers ----------
def load_sorted_frames(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png",".jpeg"))])
    out = []
    for fn in files:
        path = os.path.join(folder, fn)
        img = cv2.imread(path)
        if img is None:
            continue
        out.append((fn, img))
    return out

def moving_average(x, w):
    if w <= 1:
        return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='same')

# ---------- Core analysis ----------
def detect_flow_start_end(frames_gray):
    """
    frames_gray: list of grayscale frames (numpy arrays)
    returns: (start_idx, end_idx) inclusive indices of flow region. If not found returns (None, None)
    Algorithm:
      - compute mean absolute difference between consecutive frames
      - flow start = first index with diff > DIFF_FLOW_THRESH for DIFF_FLOW_MIN_CONSEC consecutive frames
      - flow end = last index where diff > DIFF_FLOW_THRESH for DIFF_FLOW_MIN_CONSEC consecutive frames
    """
    diffs = []
    for i in range(1, len(frames_gray)):
        d = np.mean(np.abs(frames_gray[i].astype(np.float32) - frames_gray[i-1].astype(np.float32)))
        diffs.append(d)
    diffs = np.array(diffs)
    # look for runs > thresh
    above = diffs > DIFF_FLOW_THRESH
    def first_run(arr):
        consec = 0
        for i,val in enumerate(arr):
            if val:
                consec += 1
                if consec >= DIFF_FLOW_MIN_CONSEC:
                    return i - consec + 1  # index in diffs -> corresponds to frame index start = i - consec + 1 + 1
            else:
                consec = 0
        return None
    start_diff_idx = first_run(above)
    end_diff_idx = None
    # for end, search from end
    consec = 0
    for i in range(len(above)-1, -1, -1):
        if above[i]:
            consec += 1
            if consec >= DIFF_FLOW_MIN_CONSEC:
                end_diff_idx = i + consec - 1
                break
        else:
            consec = 0
    if start_diff_idx is None:
        return None, None
    start_frame = start_diff_idx + 1
    end_frame = (end_diff_idx + 1) if end_diff_idx is not None else len(frames_gray)-1
    # safety clamps
    start_frame = max(0, start_frame)
    end_frame = min(len(frames_gray)-1, end_frame)
    if end_frame <= start_frame:
        return start_frame, start_frame
    return start_frame, end_frame

def compute_brightness_curve(frames, roi_slice):
    """
    frames: list of color frames
    roi_slice: slice tuple (y1,y2, x1,x2) on each frame
    returns: brightness list (per frame), normalized to 0..1
    We'll use V channel from HSV (value) as brightness.
    """
    bvals = []
    for _,f in frames:
        h,w = f.shape[:2]
        y1,y2,x1,x2 = roi_slice
        crop = f[y1:y2, x1:x2]
        if crop.size == 0:
            bvals.append(0.0)
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2].astype(np.float32)
        bvals.append(np.mean(v))
    bvals = np.array(bvals)
    # normalize (avoid div by 0)
    if bvals.max() - bvals.min() < 1e-6:
        norm = np.zeros_like(bvals)
    else:
        norm = (bvals - bvals.min()) / (bvals.max() - bvals.min())
    # smooth
    sm = moving_average(norm, BRIGHT_SMOOTH_WINDOW)
    return sm, bvals

def detect_blonding_point(brightness_norm, start_idx, end_idx, time_per_frame=1.0):
    """
    Find the 'blonding' point as when brightness increases rapidly near the end.
    Approach:
      - compute derivative (forward diff) of brightness_norm between start and end
      - find peak derivative after mid-shot
    Returns (blond_idx, blond_rate) as index and slope per second (approx)
    """
    if start_idx is None or end_idx is None:
        return None, 0.0
    seg = brightness_norm[start_idx:end_idx+1]
    if len(seg) < 3:
        return None, 0.0
    deriv = np.diff(seg) / (time_per_frame)
    # look for largest positive derivative in the last 60% of the shot
    cutoff = int(len(deriv) * 0.4)
    subset = deriv[cutoff:]
    if subset.size == 0:
        return None, 0.0
    rel_idx = np.argmax(subset)
    blond_idx = start_idx + cutoff + rel_idx
    blond_rate = subset[rel_idx]
    return blond_idx, float(blond_rate)

def stream_geometry_metrics(frames_gray, roi_slice):
    """
    For each frame:
      - threshold to get dark stream (adaptive thresholding)
      - compute number of stream blobs, largest blob area, centroid x
      - measure stream width at a fixed vertical slice (40% from top of ROI)
    Returns dictionaries of time series.
    """
    n = len(frames_gray)
    blob_counts = []
    largest_area = []
    centroids_x = []
    widths = []
    y1,y2,x1,x2 = roi_slice
    h_roi = max(1, y2-y1)
    # choose a horizontal row in roi to sample width (relative)
    sample_row = int(y1 + 0.45 * h_roi)
    for im in frames_gray:
        roi = im[y1:y2, x1:x2]
        if roi.size == 0:
            blob_counts.append(0); largest_area.append(0); centroids_x.append(np.nan); widths.append(0); continue
        # adaptive threshold: Otsu on inverted (stream is darker)
        _,th = cv2.threshold(roi,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # morphological clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        # find contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # filter small
        big_contours = [c for c in contours if cv2.contourArea(c) >= MIN_STREAM_AREA]
        blob_counts.append(len(big_contours))
        if big_contours:
            areas = [cv2.contourArea(c) for c in big_contours]
            largest_area.append(max(areas))
            # centroid x (relative to full frame)
            M = cv2.moments(max(big_contours, key=cv2.contourArea))
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"]) + x1
            else:
                cx = (x1+x2)//2
            centroids_x.append(cx)
        else:
            largest_area.append(0)
            centroids_x.append(np.nan)
        # width at sample_row
        row = roi[int(0.45*roi.shape[0])]
        # threshold row using same method: stream appears dark -> invert then Otsu on row
        _,row_th = cv2.threshold(row, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # measure contiguous run length in row_th
        cols = np.where(row_th > 0)[0]
        if cols.size == 0:
            widths.append(0)
        else:
            # approximate width as span of largest contiguous block
            diffs = np.diff(cols)
            breaks = np.where(diffs>1)[0]
            if breaks.size == 0:
                span = cols[-1] - cols[0] + 1
            else:
                groups = np.split(cols, breaks+1)
                spans = [g[-1]-g[0]+1 for g in groups]
                span = max(spans)
            widths.append(int(span))
    # smooth widths
    widths = moving_average(np.array(widths), STREAM_WIDTH_SMOOTH)
    return {
        "blob_counts": np.array(blob_counts),
        "largest_area": np.array(largest_area),
        "centroids_x": np.array(centroids_x),
        "widths": np.array(widths)
    }

def detect_dry_spots(frames_gray, roi_slice):
    """
    Dry spots are bright pixels that remain bright across many frames while surroundings change.
    We'll build a persistence map of pixels above DRY_SPOT_THRESH and count how many pixels are above threshold
    in > DRY_SPOT_PERSISTANCE fraction of the frames.
    Returns fraction of ROI pixels considered persistent dry spots (0..1).
    """
    y1,y2,x1,x2 = roi_slice
    h = y2-y1; w = x2-x1
    if h<=0 or w<=0:
        return 0.0
    accum = np.zeros((h,w), dtype=np.int32)
    total = 0
    for im in frames_gray:
        roi = im[y1:y2, x1:x2]
        if roi.size==0:
            continue
        mask = (roi >= DRY_SPOT_THRESH).astype(np.int32)
        accum += mask
        total += 1
    if total == 0:
        return 0.0
    persistence = accum / float(total)
    persistent_pixels = (persistence >= DRY_SPOT_PERSISTANCE).sum()
    frac = persistent_pixels / float(h*w)
    return float(frac)

# ---------- Main pipeline ----------
def analyze_shot(stab_dir=STAB_DIR, out_dir=OUT_DIR):
    frames = load_sorted_frames(stab_dir)
    if len(frames) == 0:
        raise RuntimeError("No stabilised frames found in: " + stab_dir)
    # Convert to grayscale list
    frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _,f in frames]
    H,W = frames[0][1].shape[:2]
    # define ROI as top portion of cropped image (where basket and stream appear)
    y1 = 0
    y2 = int(TOP_ROI_FRACTION * H)
    x1 = 0
    x2 = W
    roi_slice = (y1,y2,x1,x2)

    # Smooth frames a bit
    frames_gray_blur = [cv2.GaussianBlur(f, BLUR_KERNEL, 0) for f in frames_gray]

    # 1) detect flow start / end
    start_idx, end_idx = detect_flow_start_end(frames_gray_blur)
    if start_idx is None:
        # no motion detected -> maybe static shot; set to whole sequence
        start_idx = 0
        end_idx = len(frames)-1

    # time scaling: assume frames represent 1 unit per frame unless user provides FPS/time-per-frame
    time_per_frame = 1.0 * FRAME_STEP

    shot_time = (end_idx - start_idx + 1) * time_per_frame

    # 2) brightness curve
    brightness_norm, brightness_raw = compute_brightness_curve(frames, roi_slice)
    blond_idx, blond_rate = detect_blonding_point(brightness_norm, start_idx, end_idx, time_per_frame=time_per_frame)
    time_to_blond = None
    if blond_idx is not None:
        time_to_blond = (blond_idx - start_idx) * time_per_frame

    # 3) stream geometry
    geom = stream_geometry_metrics(frames_gray_blur, roi_slice)
    avg_width = float(np.nanmean(geom["widths"][start_idx:end_idx+1])) if end_idx >= start_idx else float(np.nanmean(geom["widths"]))
    width_std = float(np.nanstd(geom["widths"][start_idx:end_idx+1])) if end_idx >= start_idx else float(np.nanstd(geom["widths"]))
    multi_blob_frac = float((geom["blob_counts"][start_idx:end_idx+1] > 1).mean()) if end_idx >= start_idx else float((geom["blob_counts"]>1).mean())
    centroid_variation = float(np.nanstd(geom["centroids_x"][start_idx:end_idx+1])) if end_idx >= start_idx else float(np.nanstd(geom["centroids_x"]))

    # 4) dry spots detection
    dry_spot_frac = detect_dry_spots(frames_gray_blur[start_idx:end_idx+1], roi_slice)

    # 5) rule-based classification
    reasons = []
    # shot time check
    label = "IDEAL"
    if shot_time < MIN_IDEAL_TIME:
        label = "UNDER"
        reasons.append(f"Shot time short ({shot_time:.1f}s < {MIN_IDEAL_TIME}s)")
    elif shot_time > MAX_IDEAL_TIME:
        label = "OVER"
        reasons.append(f"Shot time long ({shot_time:.1f}s > {MAX_IDEAL_TIME}s)")

    # blonding check (fast blonding -> under/over depending on context; usually fast blonding = under)
    if time_to_blond is not None:
        blond_rate_per_s = blond_rate
        if blond_rate_per_s > FAST_BLOND_RATE:
            # fast blonding — likely under-extracted or channeling causing early blonding
            if label == "OVER":
                # contradictory signals -> list both
                reasons.append(f"Fast blonding rate ({blond_rate_per_s:.3f}/s) despite long shot time")
            else:
                label = "UNDER"
                reasons.append(f"Fast blonding ({blond_rate_per_s:.3f}/s, {time_to_blond:.1f}s after flow start)")
        else:
            # slow/normal blonding: no immediate flag
            pass
    else:
        reasons.append("Blonding not detected")

    # dry spot / channeling
    if dry_spot_frac > 0.01:  # >1% area persistent bright
        reasons.append(f"Persistent bright spots (possible channeling): {dry_spot_frac*100:.2f}% of ROI")
        # channeling often causes under-extraction (bypass)
        if label == "IDEAL":
            label = "UNDER"

    # spraying / multiple streams
    if multi_blob_frac > MULTI_BLOB_FRAC_THRESH:
        reasons.append(f"Multiple stream blobs in {multi_blob_frac*100:.1f}% frames (spraying / non-convergent flow)")
        if label == "IDEAL":
            label = "UNDER"

    # stream instability / lateral movement
    # if centroid variation is large relative to frame width -> unstable
    if not math.isnan(centroid_variation) and centroid_variation > 0.05 * W:
        reasons.append(f"High lateral stream movement (std {centroid_variation:.1f}px)")
        if label == "IDEAL":
            label = "UNDER"

    # put a final fallback reason if no reasons found and label not ideal
    if label != "IDEAL" and len(reasons) == 0:
        reasons.append("Deviation from ideal shot metrics")

    # Compose metrics
    metrics = {
        "num_frames": len(frames),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "shot_time_s": float(shot_time),
        "time_to_blond_s": float(time_to_blond) if time_to_blond is not None else None,
        "blond_rate_per_frame": float(blond_rate),
        "avg_stream_width_px": float(avg_width),
        "stream_width_std_px": float(width_std),
        "multi_blob_fraction": float(multi_blob_frac),
        "dry_spot_fraction": float(dry_spot_frac),
        "centroid_variation_px": float(centroid_variation)
    }

    # save plots
    times = np.arange(len(frames)) * time_per_frame
    plt.figure(figsize=(8,4))
    plt.plot(times, brightness_norm, label="norm brightness")
    if blond_idx is not None:
        plt.axvline(x=blond_idx * time_per_frame, color='r', linestyle='--', label='blond_point')
    plt.axvline(x=start_idx * time_per_frame, color='g', linestyle=':', label='start')
    plt.axvline(x=end_idx * time_per_frame, color='k', linestyle=':', label='end')
    plt.xlabel("time (frames)")
    plt.ylabel("normalized brightness")
    plt.legend()
    plt.title("Brightness (blonding) curve")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "brightness_curve.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(times, geom["widths"], label="stream width (px)")
    plt.plot(times, geom["largest_area"]/ (np.max(geom["largest_area"]) + 1e-8) * np.max(geom["widths"]), label="normalized area->width")
    plt.axvline(x=start_idx * time_per_frame, color='g', linestyle=':')
    plt.axvline(x=end_idx * time_per_frame, color='k', linestyle=':')
    plt.xlabel("time (frames)")
    plt.ylabel("pixels")
    plt.legend()
    plt.title("Stream width + area over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stream_width_curve.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(times, geom["blob_counts"], label="blob counts")
    plt.xlabel("time (frames)")
    plt.ylabel("count")
    plt.title("Number of stream blobs over time")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "blobs_over_time.png"))
    plt.close()

    # Save report
    report = {
        "label": label,
        "reasons": reasons,
        "metrics": metrics
    }
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    return report, metrics, {
        "brightness_norm": brightness_norm.tolist(),
        "brightness_raw": brightness_raw.tolist(),
        "geom": {k: v.tolist() for k,v in geom.items()}
    }

# ---------- Run if executed ----------
if __name__ == "__main__":
    print("Running shot analysis on stabilized frames...")
    report, metrics, ts = analyze_shot()
    print("DONE. Classification:", report["label"])
    print("Reasons:")
    for r in report["reasons"]:
        print(" -", r)
    print("Saved plots + report to:", OUT_DIR)
