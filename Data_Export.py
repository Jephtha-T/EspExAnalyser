# Export analysis JSON files to training CSV datasets.

import csv
import json
import math
import os


def _normalise_video_key(value):
    if value is None:
        return None
    text = os.path.splitext(str(value).strip().lower())[0]
    return text or None


def _video_keys_from_label_row(row):
    keys = set()

    row_id = (row.get("Id") or row.get("id") or "").strip()
    if row_id:
        try:
            numeric_id = str(int(float(row_id)))
        except ValueError:
            numeric_id = row_id.strip()
        for candidate in (numeric_id, f"test{numeric_id}", f"video_{numeric_id}"):
            normalised = _normalise_video_key(candidate)
            if normalised:
                keys.add(normalised)

    video_name = row.get("Video Name") or row.get("video_name") or ""
    normalised_name = _normalise_video_key(video_name)
    if normalised_name:
        keys.add(normalised_name)

    return keys


def load_all_results(analysis_dir):
    # Load all *_results.json files from Analysis or Analysis/results.
    results_list = []

    if not os.path.exists(analysis_dir):
        print(f"Warning: analysis directory not found: {analysis_dir}")
        return results_list

    results_subdir = os.path.join(analysis_dir, "results")
    search_dirs = [results_subdir, analysis_dir] if os.path.isdir(results_subdir) else [analysis_dir]

    loaded_paths = set()
    for current_dir in search_dirs:
        for filename in sorted(os.listdir(current_dir)):
            if not filename.endswith("_results.json"):
                continue

            filepath = os.path.join(current_dir, filename)
            if filepath in loaded_paths:
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as file_ref:
                    data = json.load(file_ref)
                    results_list.append(data)
                    loaded_paths.add(filepath)
            except Exception as err:
                print(f"Warning: could not load {filename}: {err}")

    return results_list


def normalize_value(value, min_val, max_val):
    # Min-max normalization to [0, 1].
    if max_val == min_val:
        return 0.5
    return (value - min_val) / (max_val - min_val)


def label_to_class_id(label):
    # Map label text to class id.
    if not label:
        return None

    normalized = str(label).strip().lower()

    if "under" in normalized:
        return 0
    if "ideal" in normalized:
        return 1
    if "over" in normalized:
        return 2

    return None


def load_labels(labels_csv_path):
    # Load labels from Video Data/dataset.csv.
    labels_by_video = {}
    labels_in_order = []

    if not os.path.exists(labels_csv_path):
        print(f"Warning: labels file not found: {labels_csv_path}")
        return labels_by_video, labels_in_order

    try:
        parse_error = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                labels_by_video.clear()
                labels_in_order.clear()

                with open(labels_csv_path, "r", encoding=encoding, newline="") as file_ref:
                    reader = csv.DictReader(file_ref)
                    for row in reader:
                        label_text = (row.get("Espresso Extraction Level") or "").strip()
                        label_id = label_to_class_id(label_text)
                        if label_id is None:
                            continue

                        labels_in_order.append(label_id)

                        for key in _video_keys_from_label_row(row):
                            labels_by_video[key] = label_id

                parse_error = None
                break
            except UnicodeDecodeError as err:
                parse_error = err
                continue

        if parse_error is not None:
            raise parse_error

        print(f"Loaded {len(labels_in_order)} labels from dataset.csv")
    except Exception as err:
        print(f"Warning: could not load dataset.csv: {err}")

    return labels_by_video, labels_in_order


def _as_float(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _value_at(values, index):
    if index < 0 or index >= len(values):
        return None
    return _as_float(values[index])


def _normalise_curve(values):
    valid = [value for value in values if value is not None]
    if not valid:
        return [None for _ in values]

    min_val = min(valid)
    max_val = max(valid)
    if abs(max_val - min_val) < 1e-9:
        return [0.5 if value is not None else None for value in values]

    output = []
    for value in values:
        if value is None:
            output.append(None)
        else:
            output.append((value - min_val) / (max_val - min_val))
    return output


def _get_frame_count(result):
    # Curves are measured over start+1..end, so expected count is end-start.
    flow_start = _as_int(result.get("flow_start"), 0)
    flow_end = _as_int(result.get("flow_end"), flow_start)
    expected = max(0, flow_end - flow_start)

    return max(
        expected,
        len(result.get("brightness_curve") or []),
        len(result.get("saturation_curve") or []),
        len(result.get("hue_curve") or []),
        len(result.get("blond_score_curve") or []),
        len(result.get("channeling_counts") or []),
    )


def build_frame_rows(result, label):
    video_id = str(result.get("video_name", "unknown"))
    fps = max(_as_float(result.get("fps")) or 1.0, 1.0)
    flow_start = _as_int(result.get("flow_start"), 0)
    flow_end = _as_int(result.get("flow_end"), flow_start)
    shot_time_s = max(0.0, (flow_end - flow_start + 1) / fps)

    brightness_curve = result.get("brightness_curve") or []
    saturation_curve = result.get("saturation_curve") or []
    hue_curve = result.get("hue_curve") or []
    blond_score_curve = result.get("blond_score_curve") or []
    channeling_counts = result.get("channeling_counts") or []
    channeling_norm_curve = result.get("channeling_norm_curve") or []
    channel_conf_curve = result.get("channel_detection_confidence_curve") or []
    channel_valid_curve = result.get("channel_valid_detection_curve") or []
    quadrant_counts = result.get("channel_quadrant_count_curves") or {}
    quadrant_density = result.get("channel_quadrant_density_curves") or {}
    q_top_left = quadrant_counts.get("top_left") or []
    q_top_right = quadrant_counts.get("top_right") or []
    q_bottom_left = quadrant_counts.get("bottom_left") or []
    q_bottom_right = quadrant_counts.get("bottom_right") or []
    qd_top_left = quadrant_density.get("top_left") or []
    qd_top_right = quadrant_density.get("top_right") or []
    qd_bottom_left = quadrant_density.get("bottom_left") or []
    qd_bottom_right = quadrant_density.get("bottom_right") or []
    channel_lr_asym_curve = result.get("channel_lr_asymmetry_curve") or []
    channel_tb_asym_curve = result.get("channel_tb_asymmetry_curve") or []
    channel_entropy_curve = result.get("channel_spatial_entropy_curve") or []

    frame_count = _get_frame_count(result)
    if frame_count == 0:
        return []

    ellipse = result.get("portafilter_ellipse") or {}
    channel_roi_pixels = _as_int(result.get("channel_roi_pixels"), 0)
    blond_roi_pixels = _as_int(result.get("blond_roi_pixels"), 0)
    if blond_roi_pixels <= 0:
        blond_roi_pixels = channel_roi_pixels

    if blond_score_curve:
        blond_scores = [_as_float(value) for value in blond_score_curve]
    else:
        brightness_values = [_value_at(brightness_curve, index) for index in range(frame_count)]
        blond_scores = _normalise_curve(brightness_values)

    rows = []
    for index in range(frame_count):
        brightness = _value_at(brightness_curve, index)
        saturation = _value_at(saturation_curve, index)
        hue = _value_at(hue_curve, index)
        blond_score = _value_at(blond_scores, index)
        holes = _value_at(channeling_counts, index)
        holes_norm_curve_value = _value_at(channeling_norm_curve, index)
        channel_confidence = _value_at(channel_conf_curve, index)
        valid_detection = _value_at(channel_valid_curve, index)
        q_tl = _value_at(q_top_left, index)
        q_tr = _value_at(q_top_right, index)
        q_bl = _value_at(q_bottom_left, index)
        q_br = _value_at(q_bottom_right, index)
        qd_tl = _value_at(qd_top_left, index)
        qd_tr = _value_at(qd_top_right, index)
        qd_bl = _value_at(qd_bottom_left, index)
        qd_br = _value_at(qd_bottom_right, index)
        lr_asym = _value_at(channel_lr_asym_curve, index)
        tb_asym = _value_at(channel_tb_asym_curve, index)
        spatial_entropy = _value_at(channel_entropy_curve, index)

        # Hole normalization is density-like: holes per 1000 ROI pixels.
        if holes is None:
            holes_norm = None
        elif channel_roi_pixels > 0:
            holes_norm = (holes / channel_roi_pixels) * 1000.0
        else:
            holes_norm = holes

        frame_time_s = index / fps
        norm_t = frame_time_s / shot_time_s if shot_time_s > 0 else 0.0

        rows.append(
            {
                "video_id": video_id,
                "label": label if label is not None else "",
                "frame_rel_idx": index,
                "frame_abs_idx": flow_start + 1 + index,
                "time_s": frame_time_s,
                "norm_t": norm_t,
                "fps_used": fps,
                "shot_time_s": shot_time_s,
                "roi_ellipse_cx": _as_float(ellipse.get("cx")),
                "roi_ellipse_cy": _as_float(ellipse.get("cy")),
                "roi_ellipse_w": _as_float(ellipse.get("width")),
                "roi_ellipse_h": _as_float(ellipse.get("height")),
                "roi_ellipse_angle": _as_float(ellipse.get("angle")),
                "roi_channel_px": channel_roi_pixels,
                "roi_blond_px": blond_roi_pixels,
                "brightness_mean": brightness,
                "saturation_mean": saturation,
                "hue_mean": hue,
                "blond_score": blond_score,
                "channel_holes": holes,
                "channel_holes_norm": holes_norm,
                "channel_holes_norm_curve": holes_norm_curve_value,
                "channel_detection_confidence": channel_confidence,
                "channel_valid_detection": int(valid_detection) if valid_detection is not None else None,
                "channel_q_top_left": q_tl,
                "channel_q_top_right": q_tr,
                "channel_q_bottom_left": q_bl,
                "channel_q_bottom_right": q_br,
                "channel_qd_top_left": qd_tl,
                "channel_qd_top_right": qd_tr,
                "channel_qd_bottom_left": qd_bl,
                "channel_qd_bottom_right": qd_br,
                "channel_lr_asymmetry": lr_asym,
                "channel_tb_asymmetry": tb_asym,
                "channel_spatial_entropy": spatial_entropy,
                "is_valid_frame": 1 if (brightness is not None or holes is not None) else 0,
            }
        )

    return rows


def _first_crossing(frame_rows, metric_key, threshold):
    for row in frame_rows:
        value = row.get(metric_key)
        if value is None:
            continue
        if value >= threshold:
            return row["time_s"], row["norm_t"], row["frame_abs_idx"]
    return None, None, None


def build_event_row(result, label, frame_rows):
    video_id = str(result.get("video_name", "unknown"))
    fps = max(_as_float(result.get("fps")) or 1.0, 1.0)
    flow_start = _as_int(result.get("flow_start"), 0)
    flow_end = _as_int(result.get("flow_end"), flow_start)
    shot_time_s = max(0.0, (flow_end - flow_start + 1) / fps)
    flow_detection = result.get("flow_detection") or {}
    channeling_quality = result.get("channeling_quality") or {}
    channeling_spatial_summary = result.get("channeling_spatial_summary") or {}
    channeling_temporal_summary = result.get("channeling_temporal_summary") or {}
    quality = result.get("quality") or {}

    event_row = {
        "video_id": video_id,
        "label": label if label is not None else "",
        "shot_time_s": shot_time_s,
        "fps_used": fps,
        "flow_start": flow_start,
        "flow_end": flow_end,
        "blond_frame": result.get("blond_frame"),
        "blond_rate": _as_float(result.get("blond_rate")),
        "flow_start_confidence": _as_float(flow_detection.get("start_confidence")),
        "flow_end_confidence": _as_float(flow_detection.get("end_confidence")),
        "flow_quality_score": _as_float(flow_detection.get("quality_score")),
        "channel_quality_score": _as_float(channeling_quality.get("quality_score")),
        "channel_mean_confidence": _as_float(channeling_quality.get("mean_detection_confidence")),
        "channel_valid_ratio": _as_float(channeling_quality.get("valid_detection_ratio")),
        "overall_quality_score": _as_float(quality.get("overall_score")),
        "channel_global_lr_asymmetry": _as_float(channeling_spatial_summary.get("global_left_right_asymmetry")),
        "channel_global_tb_asymmetry": _as_float(channeling_spatial_summary.get("global_top_bottom_asymmetry")),
        "channel_mean_lr_asymmetry": _as_float(channeling_spatial_summary.get("mean_lr_asymmetry")),
        "channel_mean_tb_asymmetry": _as_float(channeling_spatial_summary.get("mean_tb_asymmetry")),
        "channel_mean_spatial_entropy": _as_float(channeling_spatial_summary.get("mean_spatial_entropy")),
        "channel_peak_holes": _as_float(channeling_temporal_summary.get("peak_holes")),
        "channel_burstiness": _as_float(channeling_temporal_summary.get("burstiness")),
        "channel_early_late_holes_delta": _as_float(channeling_temporal_summary.get("early_late_holes_delta")),
    }

    for threshold in (0.2, 0.4, 0.6, 0.8):
        time_s, norm_t, frame_idx = _first_crossing(frame_rows, "blond_score", threshold)
        key_base = f"blond_{int(threshold * 100)}"
        event_row[f"t_{key_base}_s"] = time_s
        event_row[f"t_{key_base}_norm"] = norm_t
        event_row[f"f_{key_base}"] = frame_idx

    for threshold in (5.0, 10.0):
        time_s, norm_t, frame_idx = _first_crossing(frame_rows, "channel_holes_norm", threshold)
        key_base = f"channel_{int(threshold)}"
        event_row[f"t_{key_base}_s"] = time_s
        event_row[f"t_{key_base}_norm"] = norm_t
        event_row[f"f_{key_base}"] = frame_idx

    channel_rows = [row for row in frame_rows if row.get("channel_holes_norm") is not None]
    if channel_rows:
        peak_row = max(channel_rows, key=lambda row: row["channel_holes_norm"])
        event_row["channel_peak_s"] = peak_row["time_s"]
        event_row["channel_peak_norm"] = peak_row["norm_t"]
        event_row["channel_peak_value"] = peak_row["channel_holes_norm"]

        recovery_target = peak_row["channel_holes_norm"] * 0.5
        recovery_s = None
        recovery_norm = None
        for row in channel_rows:
            if row["frame_abs_idx"] <= peak_row["frame_abs_idx"]:
                continue
            if row["channel_holes_norm"] <= recovery_target:
                recovery_s = row["time_s"]
                recovery_norm = row["norm_t"]
                break
        event_row["channel_recovery_s"] = recovery_s
        event_row["channel_recovery_norm"] = recovery_norm
    else:
        event_row["channel_peak_s"] = None
        event_row["channel_peak_norm"] = None
        event_row["channel_peak_value"] = None
        event_row["channel_recovery_s"] = None
        event_row["channel_recovery_norm"] = None

    return event_row


def build_summary_row(result, label, event_row=None):
    video_id = str(result.get("video_name", "unknown"))
    channel_stats = result.get("channeling_stats") or {}
    flow_start = _as_int(result.get("flow_start"), 0)
    flow_end = _as_int(result.get("flow_end"), flow_start)

    if event_row is None:
        frame_rows = build_frame_rows(result, label)
        event_row = build_event_row(result, label, frame_rows)

    shot_time = flow_end - flow_start
    blonding_rate = _as_float(result.get("blond_rate")) or 0.0
    channeling_avg = _as_float(channel_stats.get("average")) or 0.0
    channeling_max = _as_float(channel_stats.get("max")) or 0.0
    channeling_min = _as_float(channel_stats.get("min")) or 0.0

    ch_range = channeling_max - channeling_min
    ch_range_norm = ch_range / channeling_max if channeling_max > 0 else 0.0

    if ch_range > 0:
        ch_coverage_norm = (channeling_avg - channeling_min) / ch_range
        ch_coverage_norm = max(0.0, min(1.0, ch_coverage_norm))
    else:
        ch_coverage_norm = 0.5

    return {
        "video_id": video_id,
        "label": label if label is not None else "",
        "shot_time": shot_time,
        "shot_time_s": event_row.get("shot_time_s"),
        "blond_frame": result.get("blond_frame"),
        "blonding_rate": blonding_rate,
        "blonding_rate_norm": None,
        "channeling_range": ch_range,
        "channeling_range_norm": ch_range_norm,
        "channeling_coverage_norm": ch_coverage_norm,
        "t_blond_20_s": event_row.get("t_blond_20_s"),
        "t_blond_40_s": event_row.get("t_blond_40_s"),
        "t_blond_60_s": event_row.get("t_blond_60_s"),
        "t_blond_80_s": event_row.get("t_blond_80_s"),
        "t_channel_5_s": event_row.get("t_channel_5_s"),
        "t_channel_10_s": event_row.get("t_channel_10_s"),
        "channel_peak_s": event_row.get("channel_peak_s"),
        "channel_peak_value": event_row.get("channel_peak_value"),
        "channel_recovery_s": event_row.get("channel_recovery_s"),
        "flow_quality_score": _as_float(event_row.get("flow_quality_score")) or 0.0,
        "channel_quality_score": _as_float(event_row.get("channel_quality_score")) or 0.0,
        "overall_quality_score": _as_float(event_row.get("overall_quality_score")) or 0.0,
        "channel_global_lr_asymmetry": _as_float(event_row.get("channel_global_lr_asymmetry")) or 0.0,
        "channel_global_tb_asymmetry": _as_float(event_row.get("channel_global_tb_asymmetry")) or 0.0,
        "channel_mean_spatial_entropy": _as_float(event_row.get("channel_mean_spatial_entropy")) or 0.0,
    }


def export_to_csv(analysis_dir, output_file="training_data.csv"):
    # Build frame-level and shot-level datasets from extracted result JSON files.
    print("\n" + "=" * 60)
    print("EXPORTING ANALYSIS DATA")
    print("=" * 60)

    results_list = load_all_results(analysis_dir)
    if not results_list:
        print("No results files found. Run analysis first.")
        return None

    print(f"Loaded {len(results_list)} result files")

    base_dir = os.path.dirname(analysis_dir)
    labels_csv_path = os.path.join(base_dir, "dataset.csv")
    labels_by_video, _ = load_labels(labels_csv_path)

    frame_rows_all = []
    event_rows = []
    summary_rows = []
    unmatched_video_ids = []

    for result in results_list:
        video_id = str(result.get("video_name", "unknown"))
        label = labels_by_video.get(_normalise_video_key(video_id))
        if label is None:
            unmatched_video_ids.append(video_id)

        frame_rows = build_frame_rows(result, label)
        event_row = build_event_row(result, label, frame_rows)
        summary_row = build_summary_row(result, label, event_row=event_row)

        frame_rows_all.extend(frame_rows)
        event_rows.append(event_row)
        summary_rows.append(summary_row)

    if summary_rows:
        blonding_rates = [row["blonding_rate"] for row in summary_rows]
        min_blonding = min(blonding_rates)
        max_blonding = max(blonding_rates)
        for row in summary_rows:
            row["blonding_rate_norm"] = normalize_value(
                row["blonding_rate"],
                min_blonding,
                max_blonding,
            )

    timeseries_path = os.path.join(analysis_dir, "training_data_timeseries.csv")
    events_path = os.path.join(analysis_dir, "training_data_events.csv")
    summary_path = os.path.join(analysis_dir, output_file)

    frame_fieldnames = [
        "video_id",
        "label",
        "frame_rel_idx",
        "frame_abs_idx",
        "time_s",
        "norm_t",
        "fps_used",
        "shot_time_s",
        "roi_ellipse_cx",
        "roi_ellipse_cy",
        "roi_ellipse_w",
        "roi_ellipse_h",
        "roi_ellipse_angle",
        "roi_channel_px",
        "roi_blond_px",
        "brightness_mean",
        "saturation_mean",
        "hue_mean",
        "blond_score",
        "channel_holes",
        "channel_holes_norm",
        "channel_holes_norm_curve",
        "channel_detection_confidence",
        "channel_valid_detection",
        "channel_q_top_left",
        "channel_q_top_right",
        "channel_q_bottom_left",
        "channel_q_bottom_right",
        "channel_qd_top_left",
        "channel_qd_top_right",
        "channel_qd_bottom_left",
        "channel_qd_bottom_right",
        "channel_lr_asymmetry",
        "channel_tb_asymmetry",
        "channel_spatial_entropy",
        "is_valid_frame",
    ]

    event_fieldnames = [
        "video_id",
        "label",
        "shot_time_s",
        "fps_used",
        "flow_start",
        "flow_end",
        "blond_frame",
        "blond_rate",
        "flow_start_confidence",
        "flow_end_confidence",
        "flow_quality_score",
        "channel_quality_score",
        "channel_mean_confidence",
        "channel_valid_ratio",
        "overall_quality_score",
        "channel_global_lr_asymmetry",
        "channel_global_tb_asymmetry",
        "channel_mean_lr_asymmetry",
        "channel_mean_tb_asymmetry",
        "channel_mean_spatial_entropy",
        "channel_peak_holes",
        "channel_burstiness",
        "channel_early_late_holes_delta",
        "t_blond_20_s",
        "t_blond_20_norm",
        "f_blond_20",
        "t_blond_40_s",
        "t_blond_40_norm",
        "f_blond_40",
        "t_blond_60_s",
        "t_blond_60_norm",
        "f_blond_60",
        "t_blond_80_s",
        "t_blond_80_norm",
        "f_blond_80",
        "t_channel_5_s",
        "t_channel_5_norm",
        "f_channel_5",
        "t_channel_10_s",
        "t_channel_10_norm",
        "f_channel_10",
        "channel_peak_s",
        "channel_peak_norm",
        "channel_peak_value",
        "channel_recovery_s",
        "channel_recovery_norm",
    ]

    summary_fieldnames = [
        "video_id",
        "label",
        "shot_time",
        "shot_time_s",
        "blond_frame",
        "blonding_rate",
        "blonding_rate_norm",
        "channeling_range",
        "channeling_range_norm",
        "channeling_coverage_norm",
        "t_blond_20_s",
        "t_blond_40_s",
        "t_blond_60_s",
        "t_blond_80_s",
        "t_channel_5_s",
        "t_channel_10_s",
        "channel_peak_s",
        "channel_peak_value",
        "channel_recovery_s",
        "flow_quality_score",
        "channel_quality_score",
        "overall_quality_score",
        "channel_global_lr_asymmetry",
        "channel_global_tb_asymmetry",
        "channel_mean_spatial_entropy",
    ]

    try:
        with open(timeseries_path, "w", newline="", encoding="utf-8") as file_ref:
            writer = csv.DictWriter(file_ref, fieldnames=frame_fieldnames)
            writer.writeheader()
            writer.writerows(frame_rows_all)

        with open(events_path, "w", newline="", encoding="utf-8") as file_ref:
            writer = csv.DictWriter(file_ref, fieldnames=event_fieldnames)
            writer.writeheader()
            writer.writerows(event_rows)

        with open(summary_path, "w", newline="", encoding="utf-8") as file_ref:
            writer = csv.DictWriter(file_ref, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"Saved frame-level data: {timeseries_path} ({len(frame_rows_all)} rows)")
        print(f"Saved event-level data: {events_path} ({len(event_rows)} rows)")
        print(f"Saved summary data: {summary_path} ({len(summary_rows)} rows)")
        if unmatched_video_ids:
            print(f"Warning: {len(unmatched_video_ids)} results had no label match: {sorted(set(unmatched_video_ids))}")
        print("=" * 60)

        return {
            "timeseries": timeseries_path,
            "events": events_path,
            "summary": summary_path,
        }
    except Exception as err:
        print(f"Could not write export files: {err}")
        return None


if __name__ == "__main__":
    # Example usage.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(base_dir, "Analysis")
    export_to_csv(analysis_dir, output_file="training_data.csv")
