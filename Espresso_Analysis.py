import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

from Frame_Extraction import extract_frames
from Portafilter_Detection import (
    detect_elliptical_portafilter_with_holes,
    load_image_with_orientation,
)
from Portafilter_Tracking_v2 import process_portafilter_tracking_v2
from Feature_Extraction import extract_features_from_video
from Data_Export import export_to_csv

base_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_dir, "Video Data")
image_dir = os.path.join(base_dir, "Image Data")
frames_dir = os.path.join(image_dir, "Frames")
cropped_dir = os.path.join(image_dir, "Cropped")
analysis_dir = os.path.join(base_dir, "Analysis")
analysis_blond_dir = os.path.join(analysis_dir, "blond")
analysis_channeling_dir = os.path.join(analysis_dir, "channeling")
analysis_results_dir = os.path.join(analysis_dir, "results")

video_exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")

for directory in [
    image_dir,
    frames_dir,
    cropped_dir,
    analysis_dir,
    analysis_blond_dir,
    analysis_channeling_dir,
    analysis_results_dir,
]:
    os.makedirs(directory, exist_ok=True)


def list_video_files(video_dir):
    if not os.path.isdir(video_dir):
        return []
    return [
        os.path.join(video_dir, fname)
        for fname in sorted(os.listdir(video_dir))
        if os.path.isfile(os.path.join(video_dir, fname))
        and fname.lower().endswith(video_exts)
    ]


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
        self._replay_fps = 1.0
        self._replay_start_frame = 0
        self._replay_total_frames = 0

        self.channeling_overlay_ref = None
        self.stream_mask_overlay_ref = None
        self.figure = None
        self.canvas_widget = None
        self._manual_scroll_canvas = None
        self._manual_wheel_bound = False

        self.control_frame = None
        self.log_box = None
        self.workspace = None

        self._build_start_screen()

    def _build_start_screen(self):
        self._stop_animation()
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
            threading.Thread(
                target=self._prepare_single_review,
                args=(video_path,),
                daemon=True,
            ).start()
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
            threading.Thread(target=self._run_batch, args=(videos,), daemon=True).start()

    def _prepare_single_review(self, video_path):
        try:
            # Stage 1: extract 1 FPS working frames for deterministic preview + analysis.
            clear_image_data()
            self._append_log(f"Extracting frames from: {os.path.basename(video_path)}")
            frame_count = extract_frames(video_path, frames_dir, target_fps=1)
            if frame_count < 2:
                raise RuntimeError("Not enough frames extracted for detection")

            frame_files = sorted(
                [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
            )
            if not frame_files:
                raise RuntimeError("No extracted frames found")

            first_frame_path = os.path.join(frames_dir, frame_files[0])
            second_frame_path = os.path.join(frames_dir, frame_files[min(20, len(frame_files) - 1)])

            first_frame = load_image_with_orientation(first_frame_path)
            second_frame = load_image_with_orientation(second_frame_path)

            # Stage 2: run automatic basket detection and collect debug images for review.
            _, ellipse, mode_size, fast_params, debug_images = detect_elliptical_portafilter_with_holes(
                first_frame,
                save_dashboard=False,
                use_interactive=False,
                second_frame=second_frame,
                mask_threshold=15,
                return_debug=True,
            )

            if ellipse is None:
                raise RuntimeError("Portafilter could not be detected. Try another frame or different video.")

            self.root.after(
                0,
                self._show_review_screen,
                video_path,
                frame_count,
                ellipse,
                mode_size,
                fast_params,
                debug_images,
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

        def fallback_tile(parent, label_text, tile_w, tile_h):
            holder = ttk.Frame(parent, width=tile_w, height=tile_h)
            holder.pack_propagate(False)
            ttk.Label(
                holder,
                text=label_text,
                wraplength=max(120, tile_w - 10),
                justify="center",
            ).pack(fill="both", expand=True, padx=1, pady=1)
            return holder

        def _pick_debug(*names):
            for name in names:
                value = debug_images.get(name)
                if value is not None:
                    return value
            return None

        missing = 0

        def add_tile(parent, row, title, image, img_max, is_final=False, align="center"):
            nonlocal missing
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
                missing += 1
                fallback_tile(tile, "Missing image", img_max[0], img_max[1]).pack(fill="both", expand=True)
                return

            try:
                photo, _ = get_bgr_to_tk_image(image, max_size=img_max)
            except Exception:
                photo = None
            if photo is None:
                missing += 1
                fallback_tile(tile, "Unable to render image", img_max[0], img_max[1]).pack(fill="both", expand=True)
            else:
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

        add_tile(left_col, 0, "Original", _pick_debug("Original"), side_img_max, is_final=False, align="right")
        add_tile(left_col, 1, "FAST Keypoints", debug_images.get("FAST Keypoints"), side_img_max, is_final=False, align="right")
        add_tile(
            right_col,
            0,
            "Pixel Color Change Mask",
            _pick_debug("Change Mask (Blurred)", "Change Mask"),
            side_img_max,
            is_final=False,
            align="left",
        )
        add_tile(
            right_col,
            1,
            "Detected Features and Edges",
            debug_images.get("Detected Features"),
            side_img_max,
            is_final=False,
            align="left",
        )
        add_tile(
            center_col,
            0,
            "Final Detected Portafilter",
            debug_images.get("Final Result"),
            final_img_max,
            is_final=True,
            align="center",
        )
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

        threading.Thread(
            target=self._run_full_pipeline,
            args=(
                info["video_path"],
                seed_ellipse,
                seed_mode_size,
                seed_fast_params,
            ),
            daemon=True,
        ).start()

    def _run_full_pipeline(self, video_path, manual_ellipse, manual_mode_size=None, manual_fast_params=None):
        try:
            # Stage 3: crop all frames to a stable ROI based on approved ellipse.
            tracking_result = process_portafilter_tracking_v2(
                frames_dir=frames_dir,
                output_dir=cropped_dir,
                manual_roi=False,
                manual_ellipse=manual_ellipse,
            )
            analysis_ellipse = tracking_result.get("ellipse_in_crop")
            analysis_hole_size = manual_mode_size
            analysis_fast_params = manual_fast_params
            if analysis_hole_size is None:
                analysis_hole_size = tracking_result.get("hole_mode_size")
            if analysis_fast_params is None:
                analysis_fast_params = tracking_result.get("fast_params")
            if analysis_ellipse is None:
                analysis_ellipse = manual_ellipse

            # Stage 4: run feature extraction and render results in the main Tk window.
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            feature_results = extract_features_from_video(
                cropped_frames_dir=cropped_dir,
                video_name=video_name,
                output_dir=analysis_dir,
                output_blond_dir=analysis_blond_dir,
                output_channeling_dir=analysis_channeling_dir,
                output_results_dir=analysis_results_dir,
                save_plots=True,
                detect_channeling=True,
                show_gui=False,
                capture_channeling_frames=True,
                portafilter_override_ellipse=analysis_ellipse,
                portafilter_override_hole_size=analysis_hole_size,
                portafilter_override_fast_params=analysis_fast_params,
            )

            if feature_results is None:
                raise RuntimeError("Feature extraction produced no results")

            self.root.after(0, self._show_results, video_path, tracking_result, feature_results)
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

        self.manual_state = {
            "drawing": False,
            "start": (0, 0),
            "end": (0, 0),
            "ellipse": None,
        }

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

        self._manual_scroll_canvas = tk.Canvas(viewport, highlightthickness=0)
        self._manual_scrollbar = ttk.Scrollbar(
            viewport, orient="vertical", command=self._manual_scroll_canvas.yview
        )
        self._manual_scroll_canvas.configure(yscrollcommand=self._manual_scrollbar.set)
        self._manual_scrollbar.pack(side="right", fill="y")
        self._manual_scroll_canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(self._manual_scroll_canvas)
        inner_id = self._manual_scroll_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _sync_scroll(event=None):
            self._manual_scroll_canvas.configure(
                scrollregion=self._manual_scroll_canvas.bbox("all")
            )

        inner.bind("<Configure>", _sync_scroll)
        self._manual_scroll_canvas.bind(
            "<Configure>",
            lambda event: self._manual_scroll_canvas.itemconfigure(inner_id, width=event.width),
        )
        self._manual_scroll_canvas.bind("<MouseWheel>", self._on_manual_mousewheel)
        self._manual_scroll_canvas.bind("<Button-4>", self._on_manual_mousewheel_up)
        self._manual_scroll_canvas.bind("<Button-5>", self._on_manual_mousewheel_down)
        self._bind_manual_wheel_scroll()

        self.canvas = tk.Canvas(inner, width=size[0], height=size[1], bg="#111")
        self.canvas.pack(pady=(4, 8))
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor="nw", image=photo)

        self.oval_id = None
        self.rect_id = None

        def on_button_press(event):
            self.manual_state["drawing"] = True
            self.manual_state["start"] = (event.x, event.y)
            self.manual_state["end"] = (event.x, event.y)

        def on_button_drag(event):
            if not self.manual_state["drawing"]:
                return
            self.manual_state["end"] = (event.x, event.y)
            x1, y1 = self.manual_state["start"]
            x2, y2 = self.manual_state["end"]
            if self.rect_id is not None:
                self.canvas.delete(self.rect_id)
            if self.oval_id is not None:
                self.canvas.delete(self.oval_id)
            self.rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2)
            self.oval_id = self.canvas.create_oval(x1, y1, x2, y2, outline="red", width=2)

        def on_button_release(event):
            if not self.manual_state["drawing"]:
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

        self.canvas.bind("<ButtonPress-1>", on_button_press)
        self.canvas.bind("<B1-Motion>", on_button_drag)
        self.canvas.bind("<ButtonRelease-1>", on_button_release)

        controls = ttk.Frame(inner)
        controls.pack(pady=8)

        ttk.Button(controls, text="Continue", command=self._confirm_manual_roi).pack(side="left", padx=6)
        ttk.Button(
            controls,
            text="Back to review",
            command=lambda: self._show_review_screen(
                self.preview_info.get("video_path", ""),
                len(os.listdir(frames_dir)),
                self.preview_info.get("ellipse"),
                self.preview_info.get("mode_size"),
                self.preview_info.get("fast_params"),
                self.preview_info.get("debug_images", {}),
            ),
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

        viewport = ttk.Frame(self.workspace)
        viewport.pack(fill="both", expand=True, padx=8, pady=8)

        results_canvas = tk.Canvas(viewport, highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(viewport, orient="vertical", command=results_canvas.yview)
        results_canvas.configure(yscrollcommand=results_scrollbar.set)
        results_scrollbar.pack(side="right", fill="y")
        results_canvas.pack(side="left", fill="both", expand=True)

        results_inner = ttk.Frame(results_canvas)
        inner_id = results_canvas.create_window((0, 0), window=results_inner, anchor="nw")

        def _sync_results_scroll(event=None):
            results_canvas.configure(scrollregion=results_canvas.bbox("all"))

        results_inner.bind("<Configure>", _sync_results_scroll)
        results_canvas.bind(
            "<Configure>",
            lambda event: results_canvas.itemconfigure(inner_id, width=event.width),
        )

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

        video_name = os.path.basename(video_path)
        blond_frame = feature_results.get("blond_frame")
        start_frame = feature_results.get("start_frame", 0)
        end_frame = feature_results.get("end_frame", 0)
        fps = feature_results.get("fps", 1.0) or 1.0
        shot_seconds = 0 if end_frame < start_frame else (end_frame - start_frame + 1) / fps
        channeling = feature_results.get("channeling_counts") or []
        spatial_summary = feature_results.get("channeling_spatial_summary") or {}
        channel_quality = feature_results.get("channeling_quality") or {}
        overall_quality = feature_results.get("quality") or {}

        ttk.Label(info, text=f"Video: {video_name}", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(info, text=f"Blonding frame: {blond_frame}").pack(anchor="w")
        if spatial_summary.get("dominant_quadrant"):
            ttk.Label(info, text=f"Dominant channeling quadrant: {spatial_summary['dominant_quadrant']}").pack(anchor="w")
        if spatial_summary.get("global_left_right_asymmetry") is not None:
            ttk.Label(
                info,
                text=f"Global left/right asymmetry: {float(spatial_summary.get('global_left_right_asymmetry', 0.0)):+.2f}",
            ).pack(anchor="w")
        if overall_quality.get("overall_score") is not None:
            ttk.Label(info, text=f"Overall quality score: {float(overall_quality.get('overall_score', 0.0)):.2f}").pack(anchor="w")
        quality_flags = overall_quality.get("flags") or channel_quality.get("flags") or []
        if quality_flags:
            ttk.Label(info, text=f"Quality flags: {', '.join(quality_flags)}", wraplength=620).pack(anchor="w")

        fig = Figure(figsize=(7.3, 10.2), dpi=100)
        ax_top = fig.add_subplot(2, 1, 1)
        ax_mid = fig.add_subplot(2, 1, 2)

        brightness = np.array(feature_results.get("brightness_curve"), dtype=float)
        saturation = np.array(feature_results.get("saturation_curve"), dtype=float)

        if len(brightness) > 0:
            norm_brightness = (brightness - np.nanmin(brightness)) / (np.nanmax(brightness) - np.nanmin(brightness) + 1e-6)
            if len(norm_brightness) > 2:
                norm_brightness = np.convolve(norm_brightness, np.ones(3) / 3, mode="same")

            time_axis = np.arange(len(norm_brightness)) / fps
            ax_top.plot(time_axis, norm_brightness, label="Blonding rate")
            if blond_frame is not None:
                ax_top.axvline(
                    x=(int(blond_frame) - start_frame) / fps,
                    color="r",
                    linestyle="--",
                    label="Blonding point",
                )
            ax_top.set_title("Espresso Stream Colour Transition")
            ax_top.set_xlabel("Time (s)")
            ax_top.set_ylabel("Normalised value")
            ax_top.grid(alpha=0.3)
            ax_top.legend(loc="best")
        else:
            ax_top.text(0.5, 0.5, "No brightness data", ha="center", va="center")

        if len(channeling) > 0:
            times = np.arange(len(channeling)) / fps
            channeling_arr = np.array(channeling, dtype=float)
            ax_mid.plot(times, channeling_arr, color="red", linewidth=2.0, label="Visible holes (total)")

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

                ax_mid.plot(times, q_tl, color="#1f77b4", linewidth=1.1, alpha=0.9, label="Top-left")
                ax_mid.plot(times, q_tr, color="#ff7f0e", linewidth=1.1, alpha=0.9, label="Top-right")
                ax_mid.plot(times, q_bl, color="#2ca02c", linewidth=1.1, alpha=0.9, label="Bottom-left")
                ax_mid.plot(times, q_br, color="#9467bd", linewidth=1.1, alpha=0.9, label="Bottom-right")
                ax_mid.plot(times, q_sum, color="black", linewidth=1.3, linestyle="--", label="Quadrant sum")

                max_diff = float(np.nanmax(np.abs(channeling_arr - q_sum))) if len(q_sum) > 0 else 0.0
                if max_diff > 0.5:
                    ax_mid.text(
                        0.01,
                        0.97,
                        f"Warning: max total-vs-sum diff = {max_diff:.0f}",
                        transform=ax_mid.transAxes,
                        fontsize=9,
                        va="top",
                        ha="left",
                        color="#b00020",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffe8e8", alpha=0.8),
                    )

            ax_mid.set_title("Channeling keypoints per frame (count curves)")
            ax_mid.set_xlabel("Time (s)")
            ax_mid.set_ylabel("Keypoint count")
            ax_mid.grid(alpha=0.3)
            ax_mid.fill_between(times, channeling_arr, alpha=0.3, color="red")
            ax_mid.legend(loc="best")
        else:
            ax_mid.text(0.5, 0.5, "No channeling data", ha="center", va="center")

        fig.tight_layout(pad=2.6)
        self.figure = fig
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()
        self.canvas_widget = FigureCanvasTkAgg(fig, master=left)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)

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

        shot_label = ttk.Label(
            results_inner,
            text=f"Total shot time: {shot_seconds:.2f} s",
            font=("Segoe UI", 11, "bold"),
        )
        shot_label.pack(anchor="w", pady=(8, 2), padx=10)

        btns = ttk.Frame(results_inner)
        btns.pack(fill="x", pady=10, padx=10)

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

        self._anim_job = self.root.after(250, self._animate_channeling_frame)

    def _stop_animation(self):
        if self._anim_job is not None:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None

    def _export_results(self):
        self._append_log("Exporting analysis datasets")
        exported = export_to_csv(analysis_dir, output_file="training_data.csv")
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
                clear_image_data()
                extract_frames(video_path, frames_dir, target_fps=1)
                tracking_result = process_portafilter_tracking_v2(
                    frames_dir=frames_dir,
                    output_dir=cropped_dir,
                )

                video_name = os.path.splitext(os.path.basename(video_path))[0]
                results = extract_features_from_video(
                    cropped_frames_dir=cropped_dir,
                    video_name=video_name,
                    output_dir=analysis_dir,
                    output_blond_dir=analysis_blond_dir,
                    output_channeling_dir=analysis_channeling_dir,
                    output_results_dir=analysis_results_dir,
                    save_plots=True,
                    detect_channeling=True,
                    show_gui=False,
                    capture_channeling_frames=False,
                    portafilter_override_ellipse=tracking_result.get("ellipse_in_crop"),
                    portafilter_override_hole_size=tracking_result.get("hole_mode_size"),
                    portafilter_override_fast_params=tracking_result.get("fast_params"),
                )
                if results is None:
                    raise RuntimeError("No feature results")

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
