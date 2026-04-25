import json
import os
import shutil
from dataclasses import asdict, dataclass

import cv2

from Data_Export import export_to_csv
from Feature_Extraction import extract_features_from_video
from Frame_Extraction import extract_frames, load_preview_frames
from Portafilter_Detection import (
    detect_elliptical_portafilter_with_holes,
    preload_portafilter_yolo_model,
)
from Portafilter_Tracking_v2 import process_portafilter_tracking_v2


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass
class AnalysisWorkspace:
    root_dir: str
    video_dir: str
    image_dir: str
    frames_dir: str
    cropped_dir: str
    analysis_dir: str
    analysis_blond_dir: str
    analysis_channeling_dir: str
    analysis_results_dir: str

    def ensure(self):
        for path in (
            self.root_dir,
            self.video_dir,
            self.image_dir,
            self.frames_dir,
            self.cropped_dir,
            self.analysis_dir,
            self.analysis_blond_dir,
            self.analysis_channeling_dir,
            self.analysis_results_dir,
        ):
            os.makedirs(path, exist_ok=True)


def build_workspace(root_dir, include_video_dir=True):
    root_dir = os.path.abspath(root_dir)
    video_dir = os.path.join(root_dir, "Video Data") if include_video_dir else root_dir
    image_dir = os.path.join(root_dir, "Image Data")
    analysis_dir = os.path.join(root_dir, "Analysis")
    workspace = AnalysisWorkspace(
        root_dir=root_dir,
        video_dir=video_dir,
        image_dir=image_dir,
        frames_dir=os.path.join(image_dir, "Frames"),
        cropped_dir=os.path.join(image_dir, "Cropped"),
        analysis_dir=analysis_dir,
        analysis_blond_dir=os.path.join(analysis_dir, "blond"),
        analysis_channeling_dir=os.path.join(analysis_dir, "channeling"),
        analysis_results_dir=os.path.join(analysis_dir, "results"),
    )
    workspace.ensure()
    return workspace


def clear_workspace_images(workspace):
    for folder in (workspace.frames_dir, workspace.cropped_dir):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and name.lower().endswith(VALID_IMAGE_EXTENSIONS):
                try:
                    os.remove(path)
                except OSError:
                    continue


def reset_workspace(workspace):
    for folder in (
        workspace.image_dir,
        workspace.analysis_dir,
    ):
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)
    workspace.ensure()


def _list_frame_files(frames_dir):
    if not os.path.isdir(frames_dir):
        return []
    return sorted(
        [
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if name.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]
    )


def preload_detection_models():
    return preload_portafilter_yolo_model()


def _compute_resize_scale(frame_shape, max_width=None):
    if max_width is None:
        return 1.0
    frame_height, frame_width = frame_shape[:2]
    if frame_width <= 0 or frame_width <= int(max_width):
        return 1.0
    return float(max_width) / float(frame_width)


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


def detect_portafilter_preview(
    first_frame,
    second_frame,
    *,
    preview_max_width=1280,
):
    scale = _compute_resize_scale(first_frame.shape, max_width=preview_max_width)
    inverse_scale = 1.0 / scale if scale > 0 else 1.0
    first_working = _resize_frame(first_frame, scale)
    second_working = _resize_frame(second_frame, scale) if second_frame is not None else None

    _, ellipse, mode_size, fast_params, debug_images = detect_elliptical_portafilter_with_holes(
        first_working,
        save_dashboard=False,
        use_interactive=False,
        second_frame=second_working,
        mask_threshold=15,
        return_debug=True,
    )
    if ellipse is None:
        raise RuntimeError("Portafilter could not be detected.")

    debug_images = dict(debug_images or {})
    debug_images["Original"] = first_frame
    return {
        "ellipse": _scale_ellipse(ellipse, inverse_scale),
        "mode_size": _rescale_mode_size(mode_size, inverse_scale),
        "fast_params": _rescale_fast_params(fast_params, inverse_scale),
        "debug_images": debug_images,
    }


def ellipse_to_dict(ellipse):
    if ellipse is None:
        return None
    return {
        "cx": float(ellipse[0][0]),
        "cy": float(ellipse[0][1]),
        "width": float(ellipse[1][0]),
        "height": float(ellipse[1][1]),
        "angle_deg": float(ellipse[2]),
    }


def normalize_ellipse(ellipse, frame_width, frame_height):
    if ellipse is None or frame_width <= 0 or frame_height <= 0:
        return None
    return {
        "cx": float(ellipse[0][0]) / float(frame_width),
        "cy": float(ellipse[0][1]) / float(frame_height),
        "width": float(ellipse[1][0]) / float(frame_width),
        "height": float(ellipse[1][1]) / float(frame_height),
        "angle_deg": float(ellipse[2]),
    }


def denormalize_ellipse(ellipse_payload, frame_width, frame_height):
    if not isinstance(ellipse_payload, dict):
        raise ValueError("Ellipse payload must be an object.")

    def _read_number(*keys, default=None):
        for key in keys:
            value = ellipse_payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    cx = _read_number("cx", "center_x", "centerX", "x")
    cy = _read_number("cy", "center_y", "centerY", "y")
    width = _read_number("width", "w")
    height = _read_number("height", "h")
    angle = _read_number("angle_deg", "angle", default=0.0)

    if cx is None or cy is None or width is None or height is None:
        raise ValueError("Ellipse payload is missing cx/cy/width/height.")

    values_are_normalized = (
        0.0 <= cx <= 1.0
        and 0.0 <= cy <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )
    if values_are_normalized:
        cx *= float(frame_width)
        cy *= float(frame_height)
        width *= float(frame_width)
        height *= float(frame_height)

    width = max(1.0, float(width))
    height = max(1.0, float(height))
    return ((float(cx), float(cy)), (width, height), float(angle))


def prepare_video_review(
    video_path,
    workspace,
    target_fps=1.0,
    second_frame_offset=20,
    preview_max_width=1280,
):
    workspace.ensure()
    clear_workspace_images(workspace)

    preview = load_preview_frames(
        video_path,
        target_fps=max(1.0, float(target_fps)),
        second_sample_index=second_frame_offset,
    )
    first_frame = preview["first_frame"]
    second_frame = preview["second_frame"]
    frame_count = max(
        1,
        int(
            (preview["sampling"]["total_frames"] + preview["sampling"]["frame_interval"] - 1)
            // max(1, preview["sampling"]["frame_interval"])
        ),
    )
    detection = detect_portafilter_preview(
        first_frame,
        second_frame,
        preview_max_width=preview_max_width,
    )

    frame_height, frame_width = first_frame.shape[:2]
    return {
        "video_path": video_path,
        "frame_count": int(frame_count),
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "ellipse": detection["ellipse"],
        "ellipse_dict": ellipse_to_dict(detection["ellipse"]),
        "ellipse_normalized": normalize_ellipse(detection["ellipse"], frame_width, frame_height),
        "mode_size": None if detection["mode_size"] is None else float(detection["mode_size"]),
        "fast_params": detection["fast_params"],
        "debug_images": detection["debug_images"] or {},
        "workspace": asdict(workspace),
    }


def run_full_analysis(
    video_path,
    workspace,
    approved_ellipse,
    approved_mode_size=None,
    approved_fast_params=None,
    capture_channeling_frames=True,
    progress_callback=None,
    save_crops_to_disk=True,
):
    workspace.ensure()
    clear_workspace_images(workspace)

    if progress_callback is not None:
        progress_callback(
            stage_index=4,
            stage_key="extracting_frames",
            stage_label="Frame Extraction",
            message="Extracting working frames for full analysis...",
        )
    frame_count = extract_frames(video_path, workspace.frames_dir, target_fps=1.0)
    if frame_count < 2:
        raise RuntimeError("Not enough frames extracted for analysis.")

    if progress_callback is not None:
        progress_callback(
            stage_index=5,
            stage_key="tracking_roi",
            stage_label="ROI Tracking",
            message="Tracking approved ROI across extracted frames...",
        )
    tracking_result = process_portafilter_tracking_v2(
        frames_dir=workspace.frames_dir,
        output_dir=workspace.cropped_dir,
        manual_roi=False,
        manual_ellipse=approved_ellipse,
        save_crops=save_crops_to_disk,
        return_crops=True,
    )

    analysis_ellipse = tracking_result.get("ellipse_in_crop") or approved_ellipse
    analysis_hole_size = approved_mode_size
    analysis_fast_params = approved_fast_params

    if analysis_hole_size is None:
        analysis_hole_size = tracking_result.get("hole_mode_size")
    if analysis_fast_params is None:
        analysis_fast_params = tracking_result.get("fast_params")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if progress_callback is not None:
        progress_callback(
            stage_index=6,
            stage_key="extracting_features",
            stage_label="Feature Extraction",
            message="Running espresso feature extraction and diagnostics...",
        )
    feature_results = extract_features_from_video(
        cropped_frames_dir=workspace.cropped_dir if save_crops_to_disk else None,
        video_name=video_name,
        output_dir=workspace.analysis_dir,
        output_blond_dir=workspace.analysis_blond_dir,
        output_channeling_dir=workspace.analysis_channeling_dir,
        output_results_dir=workspace.analysis_results_dir,
        save_plots=True,
        detect_channeling=True,
        show_gui=False,
        capture_channeling_frames=capture_channeling_frames,
        frames_bgr_override=tracking_result.get("cropped_frames"),
        portafilter_override_ellipse=analysis_ellipse,
        portafilter_override_hole_size=analysis_hole_size,
        portafilter_override_fast_params=analysis_fast_params,
    )
    if feature_results is None:
        raise RuntimeError("Feature extraction produced no results.")

    results_json_path = os.path.join(workspace.analysis_results_dir, f"{video_name}_results.json")
    results_json = {}
    if os.path.isfile(results_json_path):
        with open(results_json_path, "r", encoding="utf-8") as file_ref:
            results_json = json.load(file_ref)

    return {
        "video_name": video_name,
        "tracking_result": tracking_result,
        "feature_results": feature_results,
        "results_json_path": results_json_path,
        "results_json": results_json,
        "workspace": asdict(workspace),
    }


def run_batch_video(video_path, workspace):
    review = prepare_video_review(video_path, workspace)
    return run_full_analysis(
        video_path=video_path,
        workspace=workspace,
        approved_ellipse=review["ellipse"],
        approved_mode_size=review["mode_size"],
        approved_fast_params=review["fast_params"],
        capture_channeling_frames=False,
    )


def export_workspace_datasets(workspace, output_file="training_data.csv"):
    return export_to_csv(workspace.analysis_dir, output_file=output_file)
