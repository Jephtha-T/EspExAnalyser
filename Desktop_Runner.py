import os
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from Espresso_Analysis import (
    build_workspace,
    export_workspace_datasets,
    preload_detection_models,
    prepare_video_review,
    run_batch_video,
    run_full_analysis,
)
from Frame_Extraction import DEFAULT_EXTRACTION_FPS, safe_fps, shot_duration_seconds
from Portafilter_Tracking_v2 import get_locked_crop_bounds

base_dir = os.path.dirname(os.path.abspath(__file__))
workspace = build_workspace(base_dir)
video_dir = workspace.video_dir
frames_dir = workspace.frames_dir
cropped_dir = workspace.cropped_dir
analysis_dir = workspace.analysis_dir

video_exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
image_exts = (".jpg", ".jpeg", ".png")


def list_video_files(video_dir):
    if not os.path.isdir(video_dir):
        return []
    return [
        os.path.join(video_dir, fname)
        for fname in sorted(os.listdir(video_dir))
        if os.path.isfile(os.path.join(video_dir, fname))
        and fname.lower().endswith(video_exts)
    ]


def list_image_files(image_dir):
    if not os.path.isdir(image_dir):
        return []
    return [
        os.path.join(image_dir, fname)
        for fname in sorted(os.listdir(image_dir))
        if os.path.isfile(os.path.join(image_dir, fname))
        and fname.lower().endswith(image_exts)
    ]


def paste_crop_on_full_frame(full_frame, crop_frame, crop_bounds):
    if full_frame is None or crop_frame is None:
        return full_frame

    if len(crop_frame.shape) == 2 and len(full_frame.shape) == 3:
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_GRAY2BGR)

    x1, y1, x2, y2 = [int(value) for value in crop_bounds]
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    if crop_frame.shape[1] != target_w or crop_frame.shape[0] != target_h:
        crop_frame = cv2.resize(crop_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    full_h, full_w = full_frame.shape[:2]
    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(full_w, x2)
    dst_y2 = min(full_h, y2)
    if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
        return full_frame

    src_x1 = dst_x1 - x1
    src_y1 = dst_y1 - y1
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    merged = full_frame.copy()
    merged[dst_y1:dst_y2, dst_x1:dst_x2] = crop_frame[src_y1:src_y2, src_x1:src_x2]
    return merged


def project_replay_frames_to_full_frame(frames_dir, tracking_result, feature_results):
    frame_paths = list_image_files(frames_dir)
    crop_bounds = tracking_result.get("crop_bounds")
    reference_center = tracking_result.get("reference_center")
    centers = tracking_result.get("crop_centers") or []
    if not frame_paths or not crop_bounds or not reference_center or not centers:
        return False

    start_frame = int(feature_results.get("start_frame", 0) or 0)

    def project_sequence(crop_frames):
        full_frames = []
        for index, crop_frame in enumerate(crop_frames or []):
            source_index = start_frame + 1 + index
            if source_index >= len(frame_paths) or source_index >= len(centers):
                break

            full_frame = cv2.imread(frame_paths[source_index])
            if full_frame is None:
                continue

            bounds = get_locked_crop_bounds(centers[source_index], crop_bounds, reference_center)
            full_frames.append(paste_crop_on_full_frame(full_frame, crop_frame, bounds))
        return full_frames

    channeling_frames = project_sequence(feature_results.get("channeling_frames"))
    stream_mask_frames = project_sequence(feature_results.get("stream_mask_frames"))
    if channeling_frames:
        feature_results["channeling_frames"] = channeling_frames
    if stream_mask_frames:
        feature_results["stream_mask_frames"] = stream_mask_frames
    return bool(channeling_frames or stream_mask_frames)


def clear_image_data():
    for folder in [frames_dir, cropped_dir]:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)


def get_bgr_to_tk_image(image, max_size=(420, 260)):
    if image is None:
        return None, (0, 0)

    if not isinstance(image, np.ndarray):
        return None, (0, 0)

    if image.size == 0:
        return None, (0, 0)

    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(image_rgb)
    if hasattr(Image, "Resampling"):
        resample = Image.Resampling.LANCZOS
    else:
        resample = Image.ANTIALIAS

    pil_image.thumbnail(max_size, resample)
    return ImageTk.PhotoImage(pil_image), pil_image.size


class EspressoAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Espresso Analysis")
        window_width = 1200
        window_height = 1000
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        x = max(0, int((screen_width - window_width) / 2))
        y = 0
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(1200, 1000)

        self.current_mode = tk.StringVar(value="single")
        self.selected_video_path = tk.StringVar(value="")

        self.preview_info = {}
        self.preview_refs = []

        self.run_state = None
        self._anim_job = None
        self._channeling_frames = []
        self._stream_mask_frames = []
        self._anim_index = 0
        self._anim_label = None
        self._stream_mask_label = None
        self._anim_meta_label = None
        self._replay_fps = DEFAULT_EXTRACTION_FPS
        self._replay_start_frame = 0
        self._replay_total_frames = 0
        self._results_export_state = {}

        self.channeling_overlay_ref = None
        self.stream_mask_overlay_ref = None
        self.figure = None
        self.canvas_widget = None
        self._manual_scroll_canvas = None
        self._manual_wheel_bound = False
        self.canvas = None
        self.rect_id = None
        self.oval_id = None
        self.manual_state = {}
        self.manual_draw_image = None
        self.manual_scale_x = 1.0
        self.manual_scale_y = 1.0

        self.control_frame = None
        self.log_box = None
        self.workspace = None

        threading.Thread(target=preload_detection_models, daemon=True).start()
        self._build_start_screen()

    def _build_start_screen(self):
        self._stop_animation()
        self._results_export_state = {}
        if self.control_frame is not None:
            self.control_frame.destroy()
            self.control_frame = None
        if self.log_box is not None:
            self.log_box.destroy()
            self.log_box = None
        if self.workspace is not None:
            self.workspace.destroy()
            self.workspace = None

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=8)
        self.control_frame = controls

        ttk.Label(controls, text="Espresso Analysis", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )

        ttk.Label(controls, text="Mode:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=8)

        ttk.Radiobutton(
            controls,
            text="Single",
            value="single",
            variable=self.current_mode,
            command=self._on_mode_change,
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Radiobutton(
            controls,
            text="Batch (all videos in Video Data)",
            value="batch",
            variable=self.current_mode,
            command=self._on_mode_change,
        ).grid(row=1, column=2, sticky="w", padx=4)

        ttk.Button(controls, text="Browse Video", command=self._browse_video).grid(
            row=2, column=0, sticky="w", pady=8
        )
        ttk.Label(controls, textvariable=self.selected_video_path, width=95).grid(
            row=2, column=1, columnspan=3, sticky="w"
        )

        self.start_btn = ttk.Button(controls, text="Start Analysis", command=self._start)
        self.start_btn.grid(row=3, column=0, pady=10, sticky="w")

        self.close_btn = ttk.Button(controls, text="Close", command=self.root.destroy)
        self.close_btn.grid(row=3, column=1, padx=6, pady=10, sticky="w")

        self.status_var = tk.StringVar(value="Select mode and start")
        ttk.Label(controls, textvariable=self.status_var).grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(4, 0)
        )

        self.log_box = tk.Text(self.root, height=4, wrap="word")
        self.log_box.pack(fill="both", expand=False, padx=12, pady=(2, 6))
        self._append_log("Ready")

        self.workspace = ttk.Frame(self.root)
        self.workspace.pack(fill="both", expand=True, padx=10, pady=5)

        self._set_controls_enabled(True)

    def _on_mode_change(self):
        if self.current_mode.get() == "batch":
            self.selected_video_path.set("(Batch mode selected)")
        elif self.selected_video_path.get() == "(Batch mode selected)":
            self.selected_video_path.set("")

    def _set_controls_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        if self.control_frame is None:
            return
        for child in self.control_frame.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Radiobutton)):
                child.configure(state=state)

    def _browse_video(self):
        selected = filedialog.askopenfilename(
            title="Select espresso video",
            initialdir=video_dir,
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.current_mode.set("single")
            self.selected_video_path.set(selected)

    def _append_log(self, message):
        if threading.current_thread() is not threading.main_thread():
            self.root.after(0, self._append_log, message)
            return

        if self.log_box is None:
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{message}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_workspace(self):
        if self.workspace is None:
            return
        for child in self.workspace.winfo_children():
            child.destroy()
        self._unbind_manual_wheel_scroll()
        self._manual_scroll_canvas = None
        self.canvas = None

    def _on_manual_mousewheel(self, event):
        if self._manual_scroll_canvas is None:
            return
        if not self._manual_scroll_canvas.winfo_exists():
            return
        self._manual_scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_manual_mousewheel_up(self, event):
        if self._manual_scroll_canvas is None:
            return
        if not self._manual_scroll_canvas.winfo_exists():
            return
        self._manual_scroll_canvas.yview_scroll(-3, "units")

    def _on_manual_mousewheel_down(self, event):
        if self._manual_scroll_canvas is None:
            return
        if not self._manual_scroll_canvas.winfo_exists():
            return
        self._manual_scroll_canvas.yview_scroll(3, "units")

    def _bind_manual_wheel_scroll(self):
        if self._manual_wheel_bound:
            return
        self.root.bind_all("<MouseWheel>", self._on_manual_mousewheel)
        self.root.bind_all("<Button-4>", self._on_manual_mousewheel_up)
        self.root.bind_all("<Button-5>", self._on_manual_mousewheel_down)
        self._manual_wheel_bound = True

    def _unbind_manual_wheel_scroll(self):
        if not self._manual_wheel_bound:
            return
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        self._manual_wheel_bound = False

    def _start_background_task(self, target, *args):
        threading.Thread(
            target=target,
            args=args,
            daemon=True,
        ).start()

    def _create_scrollable_container(self, parent, bind_local_wheel=False):
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _sync_scroll)
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(inner_id, width=event.width),
        )

        if bind_local_wheel:
            canvas.bind("<MouseWheel>", self._on_manual_mousewheel)
            canvas.bind("<Button-4>", self._on_manual_mousewheel_up)
            canvas.bind("<Button-5>", self._on_manual_mousewheel_down)

        return canvas, scrollbar, inner

    def _create_review_fallback_tile(self, parent, label_text, tile_w, tile_h):
        holder = ttk.Frame(parent, width=tile_w, height=tile_h)
        holder.pack_propagate(False)
        ttk.Label(
            holder,
            text=label_text,
            wraplength=max(120, tile_w - 10),
            justify="center",
        ).pack(fill="both", expand=True, padx=1, pady=1)
        return holder

    def _pick_debug_image(self, debug_images, *names):
        for name in names:
            value = debug_images.get(name)
            if value is not None:
                return value
        return None

    def _add_review_tile(self, parent, row, title, image, img_max, is_final=False, align="center"):
        tile = ttk.Frame(parent)
        if align == "right":
            tile_sticky = "ne"
        elif align == "left":
            tile_sticky = "nw"
        else:
            tile_sticky = "n"
        tile.grid(row=row, column=0, sticky=tile_sticky, padx=0, pady=0)
        title_font = ("Segoe UI", 10, "bold") if is_final else ("Segoe UI", 9, "normal")
        ttk.Label(tile, text=title, font=title_font).pack(anchor="center", pady=(0, 1))

        if image is None:
            self._create_review_fallback_tile(tile, "Missing image", img_max[0], img_max[1]).pack(
                fill="both",
                expand=True,
            )
            return False

        try:
            photo, _ = get_bgr_to_tk_image(image, max_size=img_max)
        except Exception:
            photo = None

        if photo is None:
            self._create_review_fallback_tile(tile, "Unable to render image", img_max[0], img_max[1]).pack(
                fill="both",
                expand=True,
            )
            return False

        self.preview_refs.append(photo)
        image_holder = ttk.Frame(tile)
        image_holder.pack(fill="both", expand=True)
        if align == "right":
            image_anchor = "e"
        elif align == "left":
            image_anchor = "w"
        else:
            image_anchor = "center"
        ttk.Label(image_holder, image=photo).pack(anchor=image_anchor, expand=True)
        return True

    def _render_results_info(self, info, video_path, feature_results):
        summary_lines = self._build_results_summary_lines(video_path, feature_results)
        video_name = os.path.basename(video_path)
        blond_frame = feature_results.get("blond_frame")
        for idx, line in enumerate(summary_lines):
            label_kwargs = {"anchor": "w"}
            if idx == 0:
                label_kwargs["font"] = ("Segoe UI", 12, "bold")
            elif line.startswith("ML prediction:"):
                label_kwargs["font"] = ("Segoe UI", 10, "bold")
            if line.startswith("Combined assessment:") or line.startswith("Diagnostic flags:"):
                label_kwargs["wraplength"] = 620
                label_kwargs["justify"] = "left"
            ttk.Label(info, text=line, **label_kwargs).pack(anchor="w")

        return {
            "video_name": video_name,
            "blond_frame": blond_frame,
        }

    def _plot_brightness_axis(self, ax, feature_results, fps, start_frame, blond_frame):
        brightness = np.array(feature_results.get("brightness_curve"), dtype=float)
        if len(brightness) > 0:
            norm_brightness = (brightness - np.nanmin(brightness)) / (np.nanmax(brightness) - np.nanmin(brightness) + 1e-6)
            if len(norm_brightness) > 2:
                norm_brightness = np.convolve(norm_brightness, np.ones(3) / 3, mode="same")

            time_axis = np.arange(len(norm_brightness)) / fps
            ax.plot(time_axis, norm_brightness, label="Blonding rate")
            if blond_frame is not None:
                ax.axvline(
                    x=(int(blond_frame) - start_frame) / fps,
                    color="r",
                    linestyle="--",
                    label="Blonding point",
                )
            ax.set_title("Espresso Stream Colour Transition")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalised value")
            ax.grid(alpha=0.3)
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No brightness data", ha="center", va="center")

    def _plot_channeling_axis(self, ax, feature_results, fps):
        channeling = feature_results.get("channeling_counts") or []
        if len(channeling) > 0:
            times = np.arange(len(channeling)) / fps
            channeling_arr = np.array(channeling, dtype=float)
            ax.plot(times, channeling_arr, color="red", linewidth=2.0, label="Visible holes (total)")

            quadrant_curves = feature_results.get("channel_quadrant_count_curves") or {}
            q_tl = np.array(quadrant_curves.get("top_left") or [], dtype=float)
            q_tr = np.array(quadrant_curves.get("top_right") or [], dtype=float)
            q_bl = np.array(quadrant_curves.get("bottom_left") or [], dtype=float)
            q_br = np.array(quadrant_curves.get("bottom_right") or [], dtype=float)

            if (
                len(q_tl) == len(channeling_arr)
                and len(q_tr) == len(channeling_arr)
                and len(q_bl) == len(channeling_arr)
                and len(q_br) == len(channeling_arr)
            ):
                q_sum = q_tl + q_tr + q_bl + q_br

                ax.plot(times, q_tl, color="#1f77b4", linewidth=1.1, alpha=0.9, label="Top-left")
                ax.plot(times, q_tr, color="#ff7f0e", linewidth=1.1, alpha=0.9, label="Top-right")
                ax.plot(times, q_bl, color="#2ca02c", linewidth=1.1, alpha=0.9, label="Bottom-left")
                ax.plot(times, q_br, color="#9467bd", linewidth=1.1, alpha=0.9, label="Bottom-right")
                ax.plot(times, q_sum, color="black", linewidth=1.3, linestyle="--", label="Quadrant sum")

                max_diff = float(np.nanmax(np.abs(channeling_arr - q_sum))) if len(q_sum) > 0 else 0.0
                if max_diff > 0.5:
                    ax.text(
                        0.01,
                        0.97,
                        f"Warning: max total-vs-sum diff = {max_diff:.0f}",
                        transform=ax.transAxes,
                        fontsize=9,
                        va="top",
                        ha="left",
                        color="#b00020",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe8e8", alpha=0.8),
                    )

            ax.set_title("Channeling keypoints per frame (count curves)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Keypoint count")
            ax.grid(alpha=0.3)
            ax.fill_between(times, channeling_arr, alpha=0.3, color="red")
            ax.legend(loc="best")
        else:
            ax.text(0.5, 0.5, "No channeling data", ha="center", va="center")

    def _build_combined_plot_figure(self, feature_results, fps, start_frame, blond_frame):
        fig = Figure(figsize=(7.3, 10.2), dpi=100)
        ax_top = fig.add_subplot(2, 1, 1)
        ax_mid = fig.add_subplot(2, 1, 2)
        self._plot_brightness_axis(ax_top, feature_results, fps, start_frame, blond_frame)
        self._plot_channeling_axis(ax_mid, feature_results, fps)
        fig.tight_layout(pad=2.6)
        return fig

    def _build_single_chart_figure(self, chart_kind, video_path, feature_results, fps, start_frame, blond_frame):
        fig = Figure(figsize=(9.0, 5.6), dpi=140)
        ax = fig.add_subplot(1, 1, 1)
        if chart_kind == "blonding":
            self._plot_brightness_axis(ax, feature_results, fps, start_frame, blond_frame)
        else:
            self._plot_channeling_axis(ax, feature_results, fps)

        footer = self._build_chart_footer(video_path, feature_results)
        fig.text(0.02, 0.02, footer, fontsize=9, va="bottom", ha="left")
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), pad=1.8)
        return fig

    def _render_results_plots(self, parent, feature_results, fps, start_frame, blond_frame):
        fig = self._build_combined_plot_figure(feature_results, fps, start_frame, blond_frame)

        self.figure = fig
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
        self.canvas_widget = FigureCanvasTkAgg(fig, master=parent)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)

    def _build_results_summary_lines(self, video_path, feature_results):
        video_name = os.path.basename(video_path)
        blond_frame = feature_results.get("blond_frame")
        spatial_summary = feature_results.get("channeling_spatial_summary") or {}
        channel_quality = feature_results.get("channeling_quality") or {}
        overall_quality = feature_results.get("quality") or {}
        diagnostics = feature_results.get("diagnostics") or {}
        model_prediction = feature_results.get("model_prediction") or {}
        combined_assessment = feature_results.get("combined_assessment") or {}

        lines = [
            f"Video: {video_name}",
            f"Blonding frame: {blond_frame}",
        ]
        predicted_label = model_prediction.get("predicted_label")
        predicted_class = model_prediction.get("predicted_class")
        if predicted_label or predicted_class is not None:
            prediction_text = f"ML prediction: {predicted_label or 'class prediction'}"
            prediction_details = []
            if predicted_class is not None:
                prediction_details.append(f"class {predicted_class}")
            if model_prediction.get("confidence") is not None:
                prediction_details.append(f"confidence {float(model_prediction['confidence']):.2f}")
            if prediction_details:
                prediction_text += f" ({', '.join(prediction_details)})"
            lines.append(prediction_text)
        if spatial_summary.get("dominant_quadrant"):
            lines.append(f"Dominant channeling quadrant: {spatial_summary['dominant_quadrant']}")
        if spatial_summary.get("global_left_right_asymmetry") is not None:
            lines.append(
                "Global left/right asymmetry: "
                f"{float(spatial_summary.get('global_left_right_asymmetry', 0.0)):+.2f}"
            )
        if overall_quality.get("overall_score") is not None:
            lines.append(f"Overall quality score: {float(overall_quality.get('overall_score', 0.0)):.2f}")
        if combined_assessment.get("explanation"):
            lines.append(f"Combined assessment: {combined_assessment['explanation']}")

        diagnostic_flags = diagnostics.get("flags") or []
        if diagnostic_flags:
            flag_lines = []
            for flag in diagnostic_flags[:5]:
                title = flag.get("title") or flag.get("code")
                reason = flag.get("reason") or flag.get("description") or ""
                flag_lines.append(f"- {title}: {reason}")
            lines.append("Diagnostic flags:\n" + "\n".join(flag_lines))
        else:
            quality_flags = overall_quality.get("flags") or channel_quality.get("flags") or []
            if quality_flags:
                lines.append(f"Quality flags: {', '.join(quality_flags)}")
        return lines

    def _build_chart_footer(self, video_path, feature_results):
        footer_parts = [f"Video: {os.path.basename(video_path)}"]
        blond_frame = feature_results.get("blond_frame")
        if blond_frame is not None:
            footer_parts.append(f"Blonding frame: {blond_frame}")
        quality = feature_results.get("quality") or {}
        if quality.get("overall_score") is not None:
            footer_parts.append(f"Quality score: {float(quality.get('overall_score', 0.0)):.2f}")
        prediction = feature_results.get("model_prediction") or {}
        predicted_label = prediction.get("predicted_label")
        predicted_class = prediction.get("predicted_class")
        if predicted_label or predicted_class is not None:
            prediction_text = f"Prediction: {predicted_label or 'class prediction'}"
            if predicted_class is not None:
                prediction_text += f" (class {predicted_class})"
            footer_parts.append(prediction_text)
        return " | ".join(footer_parts)

    def _normalise_export_name(self, name):
        cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        cleaned = cleaned.strip("._")
        return cleaned or "espresso_analysis"

    def _save_figure(self, figure, output_path):
        figure.savefig(output_path, dpi=180, bbox_inches="tight")

    def _select_report_frame(self, frames):
        if not frames:
            return None
        return frames[min(len(frames) // 2, len(frames) - 1)]

    def _show_export_frame(self, ax, frame, title):
        ax.set_title(title)
        ax.axis("off")
        if frame is None:
            ax.text(0.5, 0.5, "No frame available", ha="center", va="center")
            return

        if len(frame.shape) == 2:
            ax.imshow(frame, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _build_results_report_figure(self, video_path, tracking_result, feature_results):
        fps = safe_fps(feature_results.get("fps"), default=DEFAULT_EXTRACTION_FPS)
        start_frame = feature_results.get("start_frame", 0)
        blond_frame = feature_results.get("blond_frame")
        report_lines = self._build_results_summary_lines(video_path, feature_results)
        tracking_count = tracking_result.get("frame_count")
        if tracking_count is not None:
            report_lines.append(f"Tracked frames: {tracking_count}")

        fig = Figure(figsize=(12.8, 15.0), dpi=140)
        grid = fig.add_gridspec(4, 2, height_ratios=[0.9, 1.2, 1.25, 1.05], hspace=0.35, wspace=0.22)
        ax_text = fig.add_subplot(grid[0, :])
        ax_brightness = fig.add_subplot(grid[1, :])
        ax_channeling = fig.add_subplot(grid[2, :])
        ax_replay = fig.add_subplot(grid[3, 0])
        ax_mask = fig.add_subplot(grid[3, 1])

        ax_text.axis("off")
        ax_text.text(
            0.0,
            1.0,
            "\n".join(report_lines),
            ha="left",
            va="top",
            fontsize=11,
            wrap=True,
        )
        self._plot_brightness_axis(ax_brightness, feature_results, fps, start_frame, blond_frame)
        self._plot_channeling_axis(ax_channeling, feature_results, fps)
        self._show_export_frame(ax_replay, self._select_report_frame(self._channeling_frames), "Replay Overlay Preview")
        self._show_export_frame(ax_mask, self._select_report_frame(self._stream_mask_frames), "Stream Mask Preview")
        fig.suptitle(f"Espresso Analysis Results: {os.path.basename(video_path)}", fontsize=16, fontweight="bold", y=0.995)
        return fig

    def _write_summary_text(self, output_path, video_path, tracking_result, feature_results):
        lines = self._build_results_summary_lines(video_path, feature_results)
        tracking_count = tracking_result.get("frame_count")
        if tracking_count is not None:
            lines.append(f"Tracked frames: {tracking_count}")
        start_frame = feature_results.get("start_frame", 0)
        end_frame = feature_results.get("end_frame", 0)
        fps = safe_fps(feature_results.get("fps"), default=DEFAULT_EXTRACTION_FPS)
        shot_seconds = shot_duration_seconds(start_frame, end_frame, fps)
        lines.append(f"Total shot time: {shot_seconds:.2f} s")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _export_frame_sequence(self, output_dir, frames, prefix, fps):
        if not frames:
            return 0, None

        os.makedirs(output_dir, exist_ok=True)
        saved_count = 0
        for idx, frame in enumerate(frames, start=1):
            frame_path = os.path.join(output_dir, f"{prefix}_{idx:04d}.png")
            if cv2.imwrite(frame_path, frame):
                saved_count += 1

        video_path = None
        first_frame = frames[0]
        if first_frame is not None and isinstance(first_frame, np.ndarray) and first_frame.size > 0:
            height, width = first_frame.shape[:2]
            if height > 0 and width > 0:
                video_path = os.path.join(output_dir, f"{prefix}_replay.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(float(fps), 0.5),
                    (width, height),
                )
                if writer.isOpened():
                    for frame in frames:
                        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                            continue
                        if frame.shape[:2] != (height, width):
                            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        writer.write(frame)
                    writer.release()
                else:
                    video_path = None
        return saved_count, video_path

    def _export_visual_results(self):
        export_state = self._results_export_state or {}
        video_path = export_state.get("video_path")
        tracking_result = export_state.get("tracking_result") or {}
        feature_results = export_state.get("feature_results") or {}
        if not video_path or not feature_results:
            messagebox.showerror("Export unavailable", "Run a single-video analysis first.")
            return

        selected_dir = filedialog.askdirectory(
            title="Choose export folder",
            initialdir=analysis_dir,
            mustexist=True,
        )
        if not selected_dir:
            return

        fps = safe_fps(feature_results.get("fps"), default=DEFAULT_EXTRACTION_FPS)
        start_frame = feature_results.get("start_frame", 0)
        blond_frame = feature_results.get("blond_frame")
        stem = self._normalise_export_name(os.path.splitext(os.path.basename(video_path))[0])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_root = os.path.join(selected_dir, f"{stem}_results_export_{timestamp}")
        charts_dir = os.path.join(export_root, "charts")
        replay_dir = os.path.join(export_root, "replay")
        channeling_dir = os.path.join(replay_dir, "channeling_frames")
        mask_dir = os.path.join(replay_dir, "stream_mask_frames")
        os.makedirs(charts_dir, exist_ok=True)
        os.makedirs(replay_dir, exist_ok=True)

        self._append_log(f"Exporting results assets to {export_root}")

        blonding_chart = self._build_single_chart_figure(
            "blonding",
            video_path,
            feature_results,
            fps,
            start_frame,
            blond_frame,
        )
        channeling_chart = self._build_single_chart_figure(
            "channeling",
            video_path,
            feature_results,
            fps,
            start_frame,
            blond_frame,
        )
        combined_chart = self._build_combined_plot_figure(feature_results, fps, start_frame, blond_frame)
        report_figure = self._build_results_report_figure(video_path, tracking_result, feature_results)

        blonding_path = os.path.join(charts_dir, "blonding_chart.png")
        channeling_path = os.path.join(charts_dir, "channeling_chart.png")
        combined_path = os.path.join(charts_dir, "combined_results_charts.png")
        report_path = os.path.join(export_root, "results_page.png")
        summary_path = os.path.join(export_root, "results_summary.txt")

        self._save_figure(blonding_chart, blonding_path)
        self._save_figure(channeling_chart, channeling_path)
        self._save_figure(combined_chart, combined_path)
        self._save_figure(report_figure, report_path)
        self._write_summary_text(summary_path, video_path, tracking_result, feature_results)

        channeling_count, channeling_video = self._export_frame_sequence(
            channeling_dir,
            self._channeling_frames,
            "channeling_overlay",
            self._replay_fps,
        )
        mask_count, mask_video = self._export_frame_sequence(
            mask_dir,
            self._stream_mask_frames,
            "stream_mask",
            self._replay_fps,
        )

        exported_lines = [
            f"Charts: {charts_dir}",
            f"Results page image: {report_path}",
            f"Summary: {summary_path}",
            f"Overlay replay frames: {channeling_count}",
            f"Mask replay frames: {mask_count}",
        ]
        if channeling_video:
            exported_lines.append(f"Overlay replay video: {channeling_video}")
        if mask_video:
            exported_lines.append(f"Mask replay video: {mask_video}")

        for line in exported_lines:
            self._append_log(line)
        messagebox.showinfo("Export complete", "Saved files:\n" + "\n".join(exported_lines))

    def _setup_results_animation(self, right, feature_results, fps, start_frame):
        vis_frame = ttk.LabelFrame(right, text="Frame-by-frame Detection")
        vis_frame.pack(fill="both", expand=True, pady=(0, 6))

        mask_frame = ttk.LabelFrame(right, text="Stream Detection Mask")
        mask_frame.pack(fill="both", expand=True)

        self._channeling_frames = feature_results.get("channeling_frames") or []
        self._stream_mask_frames = feature_results.get("stream_mask_frames") or []
        self._anim_index = 0
        self._replay_fps = float(fps)
        self._replay_start_frame = int(start_frame)
        self._replay_total_frames = max(len(self._channeling_frames), len(self._stream_mask_frames))

        self._anim_label = ttk.Label(vis_frame)
        self._anim_label.pack(padx=12, pady=12)
        self._stream_mask_label = ttk.Label(mask_frame)
        self._stream_mask_label.pack(padx=12, pady=12)
        self._anim_meta_label = ttk.Label(
            vis_frame,
            text="",
            font=("Segoe UI", 10),
        )
        self._anim_meta_label.pack(anchor="center", pady=(0, 6))
        ttk.Button(
            vis_frame,
            text="Export Results CSV",
            command=self._export_results,
        ).pack(anchor="center", pady=(0, 10))
        ttk.Button(
            vis_frame,
            text="Export Figures / Replay",
            command=self._export_visual_results,
        ).pack(anchor="center", pady=(0, 10))

    def _reset_manual_roi_state(self):
        self.manual_state = {
            "drawing": False,
            "start": (0, 0),
            "end": (0, 0),
            "ellipse": None,
        }
        self.rect_id = None
        self.oval_id = None

    def _draw_manual_roi_preview(self):
        if self.canvas is None:
            return

        x1, y1 = self.manual_state["start"]
        x2, y2 = self.manual_state["end"]

        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        if self.oval_id is not None:
            self.canvas.delete(self.oval_id)

        self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2)
        self.oval_id = self.canvas.create_oval(x1, y1, x2, y2, outline="red", width=2)

    def _on_manual_roi_press(self, event):
        self.manual_state["drawing"] = True
        self.manual_state["start"] = (event.x, event.y)
        self.manual_state["end"] = (event.x, event.y)

    def _on_manual_roi_drag(self, event):
        if not self.manual_state.get("drawing"):
            return
        self.manual_state["end"] = (event.x, event.y)
        self._draw_manual_roi_preview()

    def _on_manual_roi_release(self, event):
        if not self.manual_state.get("drawing"):
            return

        self.manual_state["drawing"] = False
        self.manual_state["end"] = (event.x, event.y)

        x1, y1 = self.manual_state["start"]
        x2, y2 = self.manual_state["end"]

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        if w < 8 or h < 8:
            self._append_log("Selection too small. Draw a larger ellipse.")
            return

        cx_o = cx * self.manual_scale_x
        cy_o = cy * self.manual_scale_y
        w_o = max(1.0, w * self.manual_scale_x)
        h_o = max(1.0, h * self.manual_scale_y)

        self.manual_state["ellipse"] = ((float(cx_o), float(cy_o)), (float(w_o), float(h_o)), 0.0)

    def _bind_manual_roi_canvas(self):
        if self.canvas is None:
            return
        self.canvas.bind("<ButtonPress-1>", self._on_manual_roi_press)
        self.canvas.bind("<B1-Motion>", self._on_manual_roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_manual_roi_release)

    def _show_review_from_preview(self):
        self._show_review_screen(
            self.preview_info.get("video_path", ""),
            self.preview_info.get("frame_count", len(os.listdir(frames_dir))),
            self.preview_info.get("ellipse"),
            self.preview_info.get("mode_size"),
            self.preview_info.get("fast_params"),
            self.preview_info.get("debug_images", {}),
        )

    def _start(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

        self._append_log("Starting...")
        mode = self.current_mode.get()

        if mode == "single":
            video_path = self.selected_video_path.get().strip()
            if not video_path:
                messagebox.showerror("No file", "Select a video file first.")
                return
            if not os.path.isfile(video_path):
                messagebox.showerror("Invalid file", f"File not found: {video_path}")
                return

            self.status_var.set("Single mode: preparing review")
            self._set_controls_enabled(False)
            self._start_background_task(self._prepare_single_review, video_path)
        else:
            videos = list_video_files(video_dir)
            if not videos:
                messagebox.showerror("No videos", f"No videos found in {video_dir}")
                return

            self.status_var.set("Batch mode: processing queue")
            self._set_controls_enabled(False)
            self._clear_workspace()
            ttk.Label(self.workspace, text="Batch processing in progress...", font=("Segoe UI", 12)).pack(
                anchor="w", padx=10, pady=10
            )
            self._start_background_task(self._run_batch, videos)

    def _prepare_single_review(self, video_path):
        try:
            self._append_log(f"Extracting frames from: {os.path.basename(video_path)}")
            review = prepare_video_review(video_path, workspace)

            self.root.after(
                0,
                self._show_review_screen,
                video_path,
                review["frame_count"],
                review["ellipse"],
                review["mode_size"],
                review["fast_params"],
                review["debug_images"],
            )
        except Exception as exc:
            self.root.after(0, self._show_error, f"Detection failed: {exc}")

    def _show_review_screen(self, video_path, frame_count, ellipse, mode_size, fast_params, debug_images):
        self._clear_workspace()
        self._stop_animation()

        if not isinstance(debug_images, dict):
            debug_images = {}

        self.preview_refs = []
        self.preview_info = {
            "video_path": video_path,
            "frame_count": frame_count,
            "ellipse": ellipse,
            "mode_size": mode_size,
            "fast_params": fast_params,
            "debug_images": debug_images,
        }

        header = ttk.Label(
            self.workspace,
            text=f"Portafilter detection review: {os.path.basename(video_path)} ({frame_count} frames)",
            font=("Segoe UI", 12, "bold"),
        )
        header.pack(anchor="w", padx=10, pady=(0, 8))

        instruction = ttk.Label(
            self.workspace,
            text="Review detection outputs. Continue with auto result, or draw manual ellipse before continuing.",
        )
        instruction.pack(anchor="w", padx=10)

        self.root.update_idletasks()
        gallery_w = max(800, self.root.winfo_width() - 12)
        gallery_h = int(max(380, self.root.winfo_height() - 320) * 0.8)
        col_w = max(220, gallery_w // 3)
        side_tile_h = max(140, (gallery_h - 8) // 2)
        side_img_max = (
            max(80, int((col_w - 6) * 0.8)),
            max(70, int(max(90, side_tile_h - 22) * 0.8)),
        )
        final_img_max = (
            max(140, int((col_w - 6) * 0.8)),
            max(140, int(max(180, gallery_h - 28) * 0.8)),
        )

        gallery = ttk.Frame(self.workspace, height=gallery_h)
        gallery.pack(fill="both", expand=True, padx=0, pady=0)
        gallery.pack_propagate(False)
        gallery.columnconfigure(0, weight=1, uniform="gallery_col")
        gallery.columnconfigure(1, weight=1, uniform="gallery_col")
        gallery.columnconfigure(2, weight=1, uniform="gallery_col")
        gallery.rowconfigure(0, weight=1)

        left_col = ttk.Frame(gallery, width=col_w, height=gallery_h)
        center_col = ttk.Frame(gallery, width=col_w, height=gallery_h)
        right_col = ttk.Frame(gallery, width=col_w, height=gallery_h)

        left_col.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        center_col.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        right_col.grid(row=0, column=2, sticky="nsew", padx=0, pady=0)

        for col in (left_col, center_col, right_col):
            col.grid_propagate(False)
            col.columnconfigure(0, weight=1)

        left_col.rowconfigure(0, weight=1)
        left_col.rowconfigure(1, weight=1)
        center_col.rowconfigure(0, weight=1)
        right_col.rowconfigure(0, weight=1)
        right_col.rowconfigure(1, weight=1)

        missing = 0

        if not self._add_review_tile(
            left_col,
            0,
            "Original",
            self._pick_debug_image(debug_images, "Original"),
            side_img_max,
            is_final=False,
            align="right",
        ):
            missing += 1
        if not self._add_review_tile(
            left_col,
            1,
            "FAST Keypoints",
            debug_images.get("FAST Keypoints"),
            side_img_max,
            is_final=False,
            align="right",
        ):
            missing += 1
        if not self._add_review_tile(
            right_col,
            0,
            "Pixel Color Change Mask",
            self._pick_debug_image(debug_images, "Change Mask (Blurred)", "Change Mask"),
            side_img_max,
            is_final=False,
            align="left",
        ):
            missing += 1
        if not self._add_review_tile(
            right_col,
            1,
            "Detected Features and Edges",
            debug_images.get("Detected Features"),
            side_img_max,
            is_final=False,
            align="left",
        ):
            missing += 1
        if not self._add_review_tile(
            center_col,
            0,
            "Final Detected Portafilter",
            debug_images.get("Final Result"),
            final_img_max,
            is_final=True,
            align="center",
        ):
            missing += 1
        if missing:
            self._append_log(f"Review payload missing {missing}/5 images.")

        btns = ttk.Frame(self.workspace)
        btns.pack(anchor="center", pady=8)

        ttk.Button(
            btns,
            text="Continue with auto ROI",
            command=lambda: self._run_analysis_with_manual_ellipse(None),
        ).pack(side="left", padx=6)
        ttk.Button(
            btns,
            text="Draw manual ROI",
            command=lambda: self._enter_manual_roi(debug_images.get("Original")),
        ).pack(side="left", padx=6)
        ttk.Button(btns, text="Back", command=self._build_start_screen).pack(side="left", padx=6)

        self.status_var.set("Review stage: choose ROI and continue")
        self._append_log("Portafilter preview loaded. Choose Continue or Draw Manual ROI.")

    def _run_analysis_with_manual_ellipse(self, manual_ellipse):
        info = self.preview_info
        if not info:
            self._show_error("Missing detection state. Start again.")
            return

        # Use the reviewed ellipse for both auto and manual runs so the final
        # analysis does not rerun portafilter detection.
        seed_ellipse = manual_ellipse
        if seed_ellipse is None:
            seed_ellipse = info["ellipse"]

        seed_mode_size = info["mode_size"] if manual_ellipse is None else None
        seed_fast_params = info["fast_params"] if manual_ellipse is None else None

        self._clear_workspace()
        self._append_log(f"Running full analysis. Manual ROI used: {manual_ellipse is not None}")

        self._start_background_task(
            self._run_full_pipeline,
            info["video_path"],
            seed_ellipse,
            seed_mode_size,
            seed_fast_params,
        )

    def _run_full_pipeline(self, video_path, manual_ellipse, manual_mode_size=None, manual_fast_params=None):
        try:
            run_output = run_full_analysis(
                video_path=video_path,
                workspace=workspace,
                approved_ellipse=manual_ellipse,
                approved_mode_size=manual_mode_size,
                approved_fast_params=manual_fast_params,
                capture_channeling_frames=True,
            )
            project_replay_frames_to_full_frame(
                workspace.frames_dir,
                run_output["tracking_result"],
                run_output["feature_results"],
            )
            self.root.after(
                0,
                self._show_results,
                video_path,
                run_output["tracking_result"],
                run_output["feature_results"],
            )
        except Exception as exc:
            self.root.after(0, self._show_error, f"Analysis failed: {exc}")

    def _enter_manual_roi(self, image):
        if image is None:
            self._show_error("Cannot open manual ROI: missing reference frame")
            return

        self._clear_workspace()
        self._stop_animation()

        self.root.update_idletasks()
        target_w = max(320, self.root.winfo_width() - 48)
        target_h = max(260, self.root.winfo_height() - 260)

        self._reset_manual_roi_state()

        title = ttk.Label(
            self.workspace,
            text="Draw an ellipse by click and drag, then click Continue.",
            font=("Segoe UI", 12, "bold"),
        )
        title.pack(anchor="w", padx=10, pady=(0, 6))

        display_image = image.copy()
        photo, size = get_bgr_to_tk_image(
            display_image,
            max_size=(target_w, target_h),
        )
        if photo is None:
            self._show_error("Failed to render manual ROI canvas")
            return

        self.manual_draw_image = image
        self.manual_scale_x = float(image.shape[1]) / float(size[0]) if size[0] else 1.0
        self.manual_scale_y = float(image.shape[0]) / float(size[1]) if size[1] else 1.0

        viewport = ttk.Frame(self.workspace)
        viewport.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self._manual_scroll_canvas, self._manual_scrollbar, inner = self._create_scrollable_container(
            viewport,
            bind_local_wheel=True,
        )
        self._bind_manual_wheel_scroll()

        self.canvas = tk.Canvas(inner, width=size[0], height=size[1], bg="#111")
        self.canvas.pack(pady=(4, 8))
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self._bind_manual_roi_canvas()

        controls = ttk.Frame(inner)
        controls.pack(pady=8)

        ttk.Button(controls, text="Continue", command=self._confirm_manual_roi).pack(side="left", padx=6)
        ttk.Button(
            controls,
            text="Back to review",
            command=self._show_review_from_preview,
        ).pack(side="left", padx=6)

        self.status_var.set("Draw manual ROI, then click Continue")

    def _confirm_manual_roi(self):
        ellipse = self.manual_state.get("ellipse")
        if ellipse is None:
            messagebox.showwarning("No ROI", "Drag on the image to draw an ellipse first.")
            return

        self._append_log(f"Manual ROI selected: {ellipse}")
        self._run_analysis_with_manual_ellipse(ellipse)

    def _show_results(self, video_path, tracking_result, feature_results):
        self._clear_workspace()
        self._append_log("Analysis complete. Rendering results UI")
        self._stop_animation()
        self._results_export_state = {
            "video_path": video_path,
            "tracking_result": tracking_result,
            "feature_results": feature_results,
        }

        viewport = ttk.Frame(self.workspace)
        viewport.pack(fill="both", expand=True, padx=8, pady=8)

        results_canvas, _results_scrollbar, results_inner = self._create_scrollable_container(viewport)
        self._manual_scroll_canvas = results_canvas
        self._bind_manual_wheel_scroll()

        top = ttk.Frame(results_inner)
        top.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        right = ttk.Frame(top)
        right.pack(side="right", fill="both", expand=True)

        info = ttk.Frame(left)
        info.pack(fill="x")

        start_frame = feature_results.get("start_frame", 0)
        end_frame = feature_results.get("end_frame", 0)
        fps = safe_fps(feature_results.get("fps"), default=DEFAULT_EXTRACTION_FPS)
        shot_seconds = shot_duration_seconds(start_frame, end_frame, fps)
        rendered_info = self._render_results_info(info, video_path, feature_results)
        blond_frame = rendered_info["blond_frame"]

        self._render_results_plots(left, feature_results, fps, start_frame, blond_frame)
        self._setup_results_animation(right, feature_results, fps, start_frame)

        shot_label = ttk.Label(
            results_inner,
            text=f"Total shot time: {shot_seconds:.2f} s",
            font=("Segoe UI", 11, "bold"),
        )
        shot_label.pack(anchor="w", pady=(8, 2), padx=10)

        btns = ttk.Frame(results_inner)
        btns.pack(fill="x", pady=10, padx=10)

        ttk.Button(btns, text="Export Figures / Replay", command=self._export_visual_results).pack(side="left", padx=6)
        ttk.Button(btns, text="Export Results CSV", command=self._export_results).pack(side="left", padx=6)
        ttk.Button(btns, text="Run another video", command=self._build_start_screen).pack(side="left", padx=6)
        ttk.Button(btns, text="Close", command=self.root.destroy).pack(side="left", padx=6)

        self.status_var.set("Analysis complete")
        self._append_log(f"Tracking frames: {tracking_result.get('frame_count')}")
        self._append_log(f"Blonding frame: {blond_frame}")
        if self._channeling_frames or self._stream_mask_frames:
            self._animate_channeling_frame()

        self._set_controls_enabled(True)

    def _animate_channeling_frame(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)

        max_len = max(len(self._channeling_frames), len(self._stream_mask_frames))
        if max_len == 0:
            if self._anim_label is not None:
                self._anim_label.configure(text="No detection preview frames")
            if self._stream_mask_label is not None:
                self._stream_mask_label.configure(text="No stream mask frames")
            return

        frame_index = self._anim_index % max_len
        self._anim_index = (self._anim_index + 1) % max_len

        if self._anim_meta_label is not None:
            rel_frame = frame_index + 1
            shot_t = (frame_index / max(1e-6, float(self._replay_fps)))
            abs_frame = self._replay_start_frame + 1 + frame_index
            self._anim_meta_label.configure(
                text=f"Replay frame {rel_frame}/{max_len} | Shot t={shot_t:.2f}s | Abs frame {abs_frame}"
            )

        if self._channeling_frames and self._anim_label is not None:
            frame = self._channeling_frames[frame_index % len(self._channeling_frames)]
            if len(frame.shape) == 2:
                frame_disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(frame_disp)
            pil_image.thumbnail((720, 420), Image.Resampling.LANCZOS)
            self.channeling_overlay_ref = ImageTk.PhotoImage(pil_image)
            self._anim_label.configure(image=self.channeling_overlay_ref)
            self._anim_label.image = self.channeling_overlay_ref
        elif self._anim_label is not None:
            self._anim_label.configure(text="No detection preview frames")

        if self._stream_mask_frames and self._stream_mask_label is not None:
            mask_frame = self._stream_mask_frames[frame_index % len(self._stream_mask_frames)]
            if len(mask_frame.shape) == 2:
                mask_disp = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2RGB)
            else:
                mask_disp = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2RGB)

            mask_image = Image.fromarray(mask_disp)
            mask_image.thumbnail((720, 420), Image.Resampling.LANCZOS)
            self.stream_mask_overlay_ref = ImageTk.PhotoImage(mask_image)
            self._stream_mask_label.configure(image=self.stream_mask_overlay_ref)
            self._stream_mask_label.image = self.stream_mask_overlay_ref
        elif self._stream_mask_label is not None:
            self._stream_mask_label.configure(text="No stream mask frames")

        replay_delay_ms = max(1, int(round(1000.0 / max(0.5, float(self._replay_fps)))))
        self._anim_job = self.root.after(replay_delay_ms, self._animate_channeling_frame)

    def _stop_animation(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    def _export_results(self):
        self._append_log("Exporting analysis datasets")
        exported = export_workspace_datasets(workspace, output_file="training_data.csv")
        if exported:
            if isinstance(exported, dict):
                lines = []
                for name in ("timeseries", "events", "summary"):
                    path = exported.get(name)
                    if path:
                        lines.append(f"{name}: {path}")
                        self._append_log(f"Exported {name}: {path}")
                messagebox.showinfo("Export complete", "Saved files:\n" + "\n".join(lines))
            else:
                messagebox.showinfo("Export complete", f"Saved to {exported}")
                self._append_log(f"Exported: {exported}")
        else:
            messagebox.showerror("Export failed", "No results found to export")

    def _run_batch(self, videos):
        total = len(videos)
        succeeded = 0
        failed = 0
        summary = []

        for idx, video_path in enumerate(videos, start=1):
            self.root.after(0, self._append_log, f"[{idx}/{total}] Running {os.path.basename(video_path)}")
            try:
                run_batch_video(video_path, workspace)
                summary.append(f"{os.path.basename(video_path)}: OK")
                succeeded += 1
            except Exception as exc:
                summary.append(f"{os.path.basename(video_path)}: FAIL ({exc})")
                failed += 1
                self.root.after(0, self._append_log, f"Failed: {video_path} - {exc}")

        self.root.after(0, self._show_batch_summary, total, succeeded, failed, summary)

    def _show_batch_summary(self, total, succeeded, failed, summary):
        self._clear_workspace()
        self._stop_animation()
        self._set_controls_enabled(True)

        title = ttk.Label(self.workspace, text="Batch Complete", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w", padx=12, pady=8)

        ttk.Label(self.workspace, text=f"Total: {total} | Success: {succeeded} | Failed: {failed}").pack(
            anchor="w", padx=12
        )

        output = tk.Text(self.workspace, height=18, wrap="word")
        output.pack(fill="both", expand=True, padx=12, pady=8)
        for row in summary:
            output.insert("end", f"{row}\n")
        output.configure(state="disabled")

        controls = ttk.Frame(self.workspace)
        controls.pack(pady=10)

        ttk.Button(controls, text="Export Results CSV", command=self._export_results).pack(side="left", padx=4)
        ttk.Button(controls, text="Back", command=self._build_start_screen).pack(side="left", padx=4)
        ttk.Button(controls, text="Close", command=self.root.destroy).pack(side="left", padx=4)

        self.status_var.set("Batch complete")
        self._append_log("Batch processing finished")

    def _show_error(self, message):
        self._set_controls_enabled(True)
        self._clear_workspace()
        self._stop_animation()

        ttk.Label(
            self.workspace,
            text="Error",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=10, pady=10)
        ttk.Label(self.workspace, text=str(message), wraplength=1200, justify="left").pack(
            anchor="w", padx=10
        )
        ttk.Button(
            self.workspace,
            text="Back",
            command=self._build_start_screen,
        ).pack(anchor="w", padx=10, pady=10)

        self.status_var.set("Error")
        self._append_log(str(message))


if __name__ == "__main__":
    root = tk.Tk()
    app = EspressoAnalysisApp(root)
    root.mainloop()
