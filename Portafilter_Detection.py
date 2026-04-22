import cv2
import os
import sys
import json
import tempfile
import subprocess
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import matplotlib.patches as mpatches

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Frame_Dir = os.path.join(Base_Dir, "Image Data/Frames")
_YOLO_MODEL_CACHE = {"path": None, "model": None, "load_error": None}


def _resolve_portafilter_model_path(model_path=None):
    if model_path:
        if os.path.isfile(model_path):
            return model_path
        return None

    candidates = [
        os.path.join(Base_Dir, "portafilter_latest.pt"),
        os.path.join(Base_Dir, "yolo_portafilter", "models", "portafilter_latest.pt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_portafilter_yolo_model(model_path=None):
    resolved = _resolve_portafilter_model_path(model_path)
    if resolved is None:
        return None, "YOLO model file not found"

    cached_path = _YOLO_MODEL_CACHE.get("path")
    cached_model = _YOLO_MODEL_CACHE.get("model")
    if cached_model is not None and cached_path == resolved:
        return cached_model, None

    try:
        from ultralytics import YOLO

        model = YOLO(resolved)
        _YOLO_MODEL_CACHE["path"] = resolved
        _YOLO_MODEL_CACHE["model"] = model
        _YOLO_MODEL_CACHE["load_error"] = None
        return model, None
    except Exception as exc:
        _YOLO_MODEL_CACHE["path"] = resolved
        _YOLO_MODEL_CACHE["model"] = None
        _YOLO_MODEL_CACHE["load_error"] = str(exc)
        return None, str(exc)


def _detect_portafilter_bbox_yolo_subprocess(image, resolved_model_path, conf=0.25, iou=0.45):
    if image is None or image.size == 0:
        return None, None, "empty image"
    if resolved_model_path is None or not os.path.isfile(resolved_model_path):
        return None, None, "model file missing"

    helper_code = (
        "import sys, json, cv2\n"
        "from ultralytics import YOLO\n"
        "img_path, model_path, conf_s, iou_s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]\n"
        "conf = float(conf_s)\n"
        "iou = float(iou_s)\n"
        "img = cv2.imread(img_path)\n"
        "if img is None:\n"
        "    print(json.dumps({'xyxy': None, 'conf': None, 'error': 'failed to read temp image'}))\n"
        "    sys.exit(0)\n"
        "model = YOLO(model_path)\n"
        "results = model.predict(source=img, conf=conf, iou=iou, verbose=False)\n"
        "if not results:\n"
        "    print(json.dumps({'xyxy': None, 'conf': None, 'error': 'no results'}))\n"
        "    sys.exit(0)\n"
        "boxes = getattr(results[0], 'boxes', None)\n"
        "if boxes is None or len(boxes) == 0:\n"
        "    print(json.dumps({'xyxy': None, 'conf': None, 'error': 'no boxes'}))\n"
        "    sys.exit(0)\n"
        "confs = boxes.conf.cpu().numpy().tolist()\n"
        "best_index = max(range(len(confs)), key=lambda idx: float(confs[idx]))\n"
        "xyxy = boxes.xyxy[best_index].cpu().numpy().tolist()\n"
        "best_conf = float(confs[best_index])\n"
        "print(json.dumps({'xyxy': xyxy, 'conf': best_conf, 'error': None}))\n"
    )

    try:
        with tempfile.TemporaryDirectory(prefix="yolo_bbox_") as temp_dir:
            temp_image_path = os.path.join(temp_dir, "frame.jpg")
            if not cv2.imwrite(temp_image_path, image):
                return None, None, "failed to write temp image"

            cmd = [
                sys.executable,
                "-c",
                helper_code,
                temp_image_path,
                resolved_model_path,
                str(float(conf)),
                str(float(iou)),
            ]
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
    except Exception as exc:
        return None, None, str(exc)

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "subprocess failed").strip()
        return None, None, detail

    raw_lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    for line in reversed(raw_lines):
        try:
            payload = json.loads(line)
        except Exception:
            continue

        xyxy = payload.get("xyxy")
        conf_score = payload.get("conf")
        if isinstance(xyxy, list) and len(xyxy) == 4 and conf_score is not None:
            return xyxy, float(conf_score), None
        error = payload.get("error")
        if error:
            return None, None, str(error)

    detail = (completed.stdout or completed.stderr or "no JSON result").strip()
    return None, None, detail


def detect_portafilter_bbox_yolo(
    image,
    model_path=None,
    conf=0.25,
    iou=0.45,
    padding_ratio=0.12,
):
    if image is None or image.size == 0:
        return None, None

    resolved = _resolve_portafilter_model_path(model_path)
    if resolved is None:
        return None, None

    xyxy = None
    best_conf = None

    model, load_error = _load_portafilter_yolo_model(model_path=resolved)
    if model is None:
        if load_error:
            print(f"YOLO preload skipped: {load_error}")
        xyxy, best_conf, sub_err = _detect_portafilter_bbox_yolo_subprocess(
            image,
            resolved_model_path=resolved,
            conf=conf,
            iou=iou,
        )
        if xyxy is None:
            if sub_err:
                print(f"YOLO subprocess fallback failed: {sub_err}")
            return None, None
        print("YOLO subprocess fallback succeeded.")
    else:
        try:
            results = model.predict(source=image, conf=float(conf), iou=float(iou), verbose=False)
        except Exception as exc:
            print(f"YOLO predict failed: {exc}")
            xyxy, best_conf, sub_err = _detect_portafilter_bbox_yolo_subprocess(
                image,
                resolved_model_path=resolved,
                conf=conf,
                iou=iou,
            )
            if xyxy is None:
                if sub_err:
                    print(f"YOLO subprocess fallback failed: {sub_err}")
                return None, None
            print("YOLO subprocess fallback succeeded.")
            results = None

        if xyxy is None:
            if not results:
                return None, None

            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return None, None

            best_index = None
            best_conf = -1.0
            try:
                confs = boxes.conf.cpu().numpy().tolist()
            except Exception:
                confs = []

            for index, score in enumerate(confs):
                score = float(score)
                if score > best_conf:
                    best_conf = score
                    best_index = index

            if best_index is None:
                return None, None

            try:
                xyxy = boxes.xyxy[best_index].cpu().numpy().tolist()
            except Exception:
                return None, None

    if len(xyxy) != 4:
        return None, None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * float(padding_ratio)
    pad_y = box_h * float(padding_ratio)

    x1 = max(0, int(np.floor(x1 - pad_x)))
    y1 = max(0, int(np.floor(y1 - pad_y)))
    x2 = min(w, int(np.ceil(x2 + pad_x)))
    y2 = min(h, int(np.ceil(y2 + pad_y)))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None

    return (x1, y1, x2, y2), float(best_conf)


def _offset_ellipse(ellipse, offset):
    if ellipse is None:
        return None

    ox, oy = offset
    (cx, cy), axes, angle = ellipse
    return ((cx + ox, cy + oy), axes, angle)


def _ellipse_from_roi_bounds(image_shape, pad_ratio_x=0.08, pad_ratio_y=0.10):
    if image_shape is None or len(image_shape) < 2:
        return None

    h, w = int(image_shape[0]), int(image_shape[1])
    if h <= 2 or w <= 2:
        return None

    pad_x = max(2, int(round(w * float(pad_ratio_x))))
    pad_y = max(2, int(round(h * float(pad_ratio_y))))

    left = min(max(0, pad_x), max(0, w - 2))
    right = max(left + 2, min(w, w - pad_x))
    top = min(max(0, pad_y), max(0, h - 2))
    bottom = max(top + 2, min(h, h - pad_y))

    major = float(max(10, right - left))
    minor = float(max(10, bottom - top))
    angle = 90.0 if minor >= major else 0.0

    return _normalize_ellipse(((w / 2.0, h / 2.0), (major, minor), angle))


def find_mode_size(keypoint_sizes, tolerance=5):
    if not keypoint_sizes:
        return None

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

    if not size_groups:
        return None

    return max(size_groups, key=lambda value: size_groups[value])


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
            # Fallback: keep FAST points when local contour quality check is unstable.
            filtered_keypoints.append(keypoint)
            keypoint_sizes.append(size)

    return filtered_keypoints, keypoint_sizes

def create_interactive_dashboard(images_dict):
    # Convert dictionary to list for easier navigation
    step_names = list(images_dict.keys())
    images = list(images_dict.values())
    current_index = 0
    
    # Create figure for slideshow with more space for buttons
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.suptitle('Portafilter Detection Pipeline - Slideshow', fontsize=16, fontweight='bold')
    
    # Enable zoom and pan functionality
    ax.set_navigate(True)
    
    def update_slide():
        # Update the current slide.
        ax.clear()
        
        # Get current image and name
        img = images[current_index]
        step_name = step_names[current_index]
        
        # Convert image to RGB if needed
        if len(img.shape) == 3 and img.shape[2] == 3:
            # BGR to RGB conversion for OpenCV images
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            # Grayscale to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        
        # Display image
        ax.imshow(img_rgb)
        ax.set_title(f'{step_name} ({current_index + 1}/{len(images)})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Update navigation info
        nav_text.set_text(f'Step {current_index + 1} of {len(images)}: {step_name}')
        
        plt.draw()
    
    def next_slide(event):
        # Go to the next slide.
        nonlocal current_index
        current_index = (current_index + 1) % len(images)
        update_slide()
    
    def prev_slide(event):
        # Go to the previous slide.
        nonlocal current_index
        current_index = (current_index - 1) % len(images)
        update_slide()
    
    # Create smaller navigation buttons positioned to avoid image overlap
    button_height = 0.04
    button_width = 0.08
    
    # Previous button - bottom left
    prev_ax = plt.axes((0.05, 0.02, button_width, button_height))
    prev_button = Button(prev_ax, '← Previous', color='lightgray', hovercolor='lightblue')
    prev_button.on_clicked(prev_slide)
    
    # Next button - bottom right
    next_ax = plt.axes((0.87, 0.02, button_width, button_height))
    next_button = Button(next_ax, 'Next →', color='lightgray', hovercolor='lightblue')
    next_button.on_clicked(next_slide)
    
    # Navigation info text - top left
    nav_text = fig.text(0.02, 0.95, '', fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Instructions text - top right
    fig.text(0.98, 0.95, 'Navigation: ← → arrow keys\n'
             'Zoom: Mouse wheel\n'
             'Pan: Click and drag', 
             fontsize=8, style='italic', ha='right', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Keyboard navigation
    def on_key(event):
        if event.key == 'left':
            prev_slide(None)
        elif event.key == 'right':
            next_slide(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initialize first slide
    update_slide()
    
    # Enable navigation toolbar
    plt.rcParams['toolbar'] = 'toolbar2'
    
    plt.show()
    
    return fig

def create_debug_dashboard(images_dict, window_name="Portafilter Detection Pipeline", save_path=None):
    # Define the grid layout
    grid_rows = 3
    grid_cols = 4
    cell_width = 400
    cell_height = 300
    
    # Create the dashboard image
    dashboard = np.zeros((cell_height * grid_rows, cell_width * grid_cols, 3), dtype=np.uint8)
    
    # Add title
    cv2.putText(dashboard, "Portafilter Detection Pipeline", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Place images in grid
    for idx, (step_name, img) in enumerate(images_dict.items()):
        if img is None:
            continue
            
        row = idx // grid_cols
        col = idx % grid_cols
        
        if row >= grid_rows:
            break
            
        # Resize image to fit cell
        if len(img.shape) == 3:
            h, w = img.shape[:2]
        else:
            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Calculate scaling to fit in cell
        scale = min(cell_width * 0.9 / w, cell_height * 0.8 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        if new_w > 0 and new_h > 0:
            resized_img = cv2.resize(img, (new_w, new_h))
            
            # Calculate position to center in cell
            y_start = row * cell_height + 50
            x_start = col * cell_width + (cell_width - new_w) // 2
            y_end = y_start + new_h
            x_end = x_start + new_w
            
            # Place image in dashboard
            dashboard[y_start:y_end, x_start:x_end] = resized_img
            
            # Add step name
            cv2.putText(dashboard, step_name, (col * cell_width + 10, row * cell_height + 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show dashboard
    cv2.imshow(window_name, dashboard)
    
    # Save dashboard if path provided
    if save_path:
        cv2.imwrite(save_path, dashboard)
        print(f"Dashboard saved to: {save_path}")
    
    return dashboard

def load_image_with_orientation(path):
    image = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image.getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print("EXIF orientation not found or failed:", e)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def safe_draw_ellipse(img, ellipse, color, thickness=1):
    if ellipse is None:
        return
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return
    if not (np.isfinite(ellipse[1][0]) and np.isfinite(ellipse[1][1])):
        return
    cv2.ellipse(img, ellipse, color, thickness)


def _is_similar_ellipse(e1, e2, center_tol=10.0, axis_tol=12.0, angle_tol=14.0):
    if e1 is None or e2 is None:
        return False

    (c1x, c1y), (a1x, a1y), ang1 = e1
    (c2x, c2y), (a2x, a2y), ang2 = e2

    center_dist = np.hypot(float(c1x) - float(c2x), float(c1y) - float(c2y))
    if center_dist > float(center_tol):
        return False

    axes_1 = sorted([float(a1x), float(a1y)])
    axes_2 = sorted([float(a2x), float(a2y)])
    if abs(axes_1[0] - axes_2[0]) > float(axis_tol) or abs(axes_1[1] - axes_2[1]) > float(axis_tol):
        return False

    delta_angle = abs(float(ang1) - float(ang2)) % 180.0
    delta_angle = min(delta_angle, 180.0 - delta_angle)
    return delta_angle <= float(angle_tol)


def _append_unique_ellipse(ellipses, areas, ellipse, area):
    for existing in ellipses:
        if _is_similar_ellipse(existing, ellipse):
            return False
    ellipses.append(ellipse)
    areas.append(float(area))
    return True


def _extract_ellipse_candidates_from_binary(
    binary_image,
    min_area,
    min_axis,
    max_aspect=8.0,
    retrieval_modes=(cv2.RETR_LIST, cv2.RETR_EXTERNAL),
):
    if binary_image is None or binary_image.size == 0:
        return [], []

    candidates = []
    candidate_areas = []

    for retrieval_mode in retrieval_modes:
        try:
            contours, _ = cv2.findContours(binary_image, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            continue

        for cnt in contours:
            if cnt is None or len(cnt) < 5:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
            except Exception:
                continue

            x_len, y_len = ellipse[1]
            if not (np.isfinite(x_len) and np.isfinite(y_len)):
                continue
            if x_len <= 0 or y_len <= 0:
                continue
            if min(x_len, y_len) < float(min_axis):
                continue

            aspect_ratio = max(x_len, y_len) / max(1e-6, min(x_len, y_len))
            if aspect_ratio > float(max_aspect):
                continue

            area = float(x_len * y_len * np.pi)
            if area < float(min_area):
                continue

            _append_unique_ellipse(candidates, candidate_areas, ellipse, area)

    return candidates, candidate_areas


def count_keypoints_in_ellipse(ellipse, keypoints, image_shape):
    if ellipse is None or keypoints is None or len(keypoints) == 0:
        return 0

    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, 255, -1)

    count = 0
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < w and 0 <= y < h and mask[y, x] == 255:
            count += 1
    return count


def _normalize_ellipse(ellipse):
    if ellipse is None or len(ellipse) != 3:
        return None

    (cx, cy), (major, minor), angle = ellipse
    values = [cx, cy, major, minor, angle]
    if not all(np.isfinite(v) for v in values):
        return None
    if major <= 1 or minor <= 1:
        return None

    return ((float(cx), float(cy)), (float(major), float(minor)), float(angle))


def select_manual_ellipse(image, window_name="Manual Portafilter ROI"):
    # Manual ellipse selector controls:
    # click+drag = draw, Enter/Space = confirm, r = reset, Esc = cancel
    if image is None:
        return None

    display = image.copy()
    drag_state = {"drawing": False, "start": None, "end": None, "ellipse": None}

    def _draw_overlay(frame):
        canvas = frame.copy()

        if drag_state["start"] is not None and drag_state["end"] is not None:
            x1, y1 = drag_state["start"]
            x2, y2 = drag_state["end"]
            left, right = sorted([x1, x2])
            top, bottom = sorted([y1, y2])

            w = right - left
            h = bottom - top
            if w > 1 and h > 1:
                ellipse = (
                    (left + w / 2.0, top + h / 2.0),
                    (float(w), float(h)),
                    0.0,
                )
                drag_state["ellipse"] = _normalize_ellipse(ellipse)
                cv2.rectangle(canvas, (left, top), (right, bottom), (0, 255, 255), 1)
                safe_draw_ellipse(canvas, drag_state["ellipse"], (0, 255, 0), 2)

        cv2.putText(
            canvas,
            "Drag mouse to draw ellipse | Enter=confirm | r=reset | Esc=cancel",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            canvas,
            "Tip: include the full basket rim and some margin.",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )
        return canvas

    def _on_mouse(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_state["drawing"] = True
            drag_state["start"] = (x, y)
            drag_state["end"] = (x, y)
            drag_state["ellipse"] = None
        elif event == cv2.EVENT_MOUSEMOVE and drag_state["drawing"]:
            drag_state["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_state["drawing"]:
            drag_state["drawing"] = False
            drag_state["end"] = (x, y)

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, _on_mouse)
    except Exception as e:
        print(f"Could not open manual ROI window: {e}")
        return None

    selected = None
    while True:
        canvas = _draw_overlay(display)
        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 32):  # Enter or Space
            if drag_state["ellipse"] is not None:
                selected = drag_state["ellipse"]
                break
            print("No ellipse selected yet. Drag to create one first.")
        elif key in (ord('r'), ord('R')):
            drag_state = {"drawing": False, "start": None, "end": None, "ellipse": None}
        elif key == 27:  # Esc
            break

    cv2.destroyWindow(window_name)
    return selected

def ellipse_feature_score(image, ellipses, areas, fast_keypoints=None, min_area=500):
    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    best_ellipse = None
    best_score = float('-inf')
    # Count loop index is implicit from enumerate.
    print(f"Detected {len(ellipses)} candidate ellipses")

    for index, el in enumerate(ellipses):
        # Basic ellipse info
        (cx, cy), (x_len, y_len), angle = el
        if not (np.isfinite(x_len) and np.isfinite(y_len)):
            continue
        if x_len <= 0 or y_len <= 0:
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, el, (255, 255, 255), -1)

        area = np.count_nonzero(mask)
        area = areas[index] if index < len(areas) else area  # Use precomputed area
        if area < min_area:
            continue

        # Calculate overall feature density score (FAST keypoints only)
        density_score = calculate_feature_density_score(el, image.shape, fast_keypoints)
        
        # Use density score as the primary scoring metric
        feature_score = density_score

        if fast_keypoints is None:
            continue

        if feature_score < 5:
            continue

        # Penalize off-center ellipses
        dist_from_center = np.linalg.norm(np.array([cx, cy]) - image_center)
        center_penalty = dist_from_center * 0.02

        # Final score: density-based with center bias
        score = feature_score - center_penalty
        #print(f"Ellipse: {el}, Score: {score:.2f}, Area: {area}, Features: {feature_score:.2f}")

        if score > best_score:
            best_score = score
            best_ellipse = el
    print(f"Best Ellipse Score: {best_score:.2f}")
    return best_ellipse

def crop_image_by_ellipse(image, ellipse, padding=10):
    if ellipse is None:
        print("No ellipse provided for cropping.")
        return image  # Return original image as fallback

    (cx, cy), (major, minor), angle = ellipse

    # Validate ellipse parameters
    if not (np.isfinite(cx) and np.isfinite(cy) and
            np.isfinite(major) and np.isfinite(minor) and
            major > 0 and minor > 0):
        print("Invalid ellipse parameters.")
        return image

    height, width = image.shape[:2]

    # Compute ellipse bounding box using cv2.boundingRect on ellipse contour
    mask = np.zeros((height, width), dtype=np.uint8)
    try:
        cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
    except Exception as e:
        print("Failed to draw ellipse:", e)
        return image

    # Find bounding box from mask
    ys, xs = np.where(mask == 255)
    if len(xs) == 0 or len(ys) == 0:
        print("No ellipse pixels found in mask.")
        return image

    x1 = max(int(xs.min()) - padding, 0)
    x2 = min(int(xs.max()) + padding, width)
    y1 = max(int(ys.min()) - padding, 0)
    y2 = height

    cropped = image[y1:y2, x1:x2]
    return cropped





def calculate_feature_density_score(ellipse, image_shape, fast_keypoints):
    if ellipse is None:
        return 0
    
    (cx, cy), (major, minor), angle = ellipse
    height, width = image_shape[:2]
    
    # Create ellipse mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
    
    # Calculate ellipse area
    ellipse_area = np.count_nonzero(mask)
    
    if ellipse_area == 0:
        return 0
    
    # Count FAST keypoints within ellipse (primary feature)
    feature_count = 0
    if fast_keypoints is not None:
        circle_count = 0
        for kp in fast_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                circle_count += 1
        feature_count += circle_count
    
    # Calculate density (features per pixel)
    density = (feature_count**2) / (ellipse_area/5)

    #print(f"  Density Analysis:")
    #print(f"    - Ellipse area: {ellipse_area} pixels")
    #print(f"    - Total features: {feature_count}")
    #print(f"    - Feature density: {density:.6f} features/pixel")
    
    return density

def generate_change_mask(frame1, frame2, threshold=30):
    # Ensure both frames are the same size
    frame1 = cv2.resize(frame1, (frame2.shape[1], frame2.shape[0]))

    # Convert to grayscale or work in LAB space for better colour comparison
    lab1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2LAB)

    # Compute absolute difference
    diff = cv2.absdiff(lab1, lab2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to get significant colour changes
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    # Optional: clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask

def detect_elliptical_portafilter_with_holes(
    image,
    save_dashboard=False,
    dashboard_path=None,
    use_interactive=True,
    second_frame=None,
    mask_threshold=30,
    manual_roi=False,
    manual_ellipse=None,
    use_yolo_prefilter=True,
    yolo_model_path=None,
    yolo_conf=0.25,
    yolo_iou=0.45,
    yolo_padding_ratio=0.12,
    return_debug=False,
):
    if image is None:
        print(f"Error loading image: {image}")
        return None
    
    original_image = image.copy()

    # Dictionary to store all intermediary steps for dashboard
    debug_images = {}
    debug_images["Original"] = original_image.copy()

    manual_ellipse = _normalize_ellipse(manual_ellipse)

    if manual_roi and manual_ellipse is None:
        print("Manual ROI mode enabled. Please select the portafilter ellipse.")
        manual_ellipse = select_manual_ellipse(original_image)
        if manual_ellipse is None:
            print("Manual ROI selection cancelled; falling back to automatic detection.")

    roi_offset = (0, 0)
    yolo_bbox = None
    if use_yolo_prefilter and manual_ellipse is None and not manual_roi:
        yolo_bbox, yolo_score = detect_portafilter_bbox_yolo(
            original_image,
            model_path=yolo_model_path,
            conf=yolo_conf,
            iou=yolo_iou,
            padding_ratio=yolo_padding_ratio,
        )
        if yolo_bbox is not None:
            x1, y1, x2, y2 = yolo_bbox
            roi_offset = (x1, y1)
            image = original_image[y1:y2, x1:x2].copy()
            if second_frame is not None:
                if second_frame.shape[0] >= y2 and second_frame.shape[1] >= x2:
                    second_frame = second_frame[y1:y2, x1:x2].copy()
                else:
                    second_frame = None
                    print("Second frame is smaller than YOLO ROI bounds; skipping change-mask assist.")

            yolo_overlay = original_image.copy()
            cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                yolo_overlay,
                f"YOLO ROI conf={yolo_score:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            debug_images["YOLO ROI"] = yolo_overlay
            print(f"Using YOLO prefilter ROI: {(x1, y1, x2, y2)} (conf={yolo_score:.3f})")
        else:
            print("YOLO prefilter not available or no detection found; using full frame.")

    output = image.copy()
    output2 = image.copy()
    debug_output = image.copy()  # For showing FAST evaluation process

    # Check if image is too sharp or too blurry
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(f"Laplacian variance: {lap_var:.2f}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_h, frame_w = gray.shape[:2]
    frame_area = float(frame_h * frame_w)

    # Scale ellipse filters to the current frame/ROI size so cropped YOLO regions
    # and full-frame fallback both behave consistently.
    min_area_auto = max(1200.0, frame_area * 0.025)
    min_area_mask = max(900.0, frame_area * 0.012)
    min_axis_len = max(10.0, min(frame_h, frame_w) * 0.05)

    #  Change Mask between first frame (image) and provided second_frame 
    if second_frame is not None:
        try:
            change_mask = generate_change_mask(image, second_frame, threshold=mask_threshold)
            change_mask_blurred = cv2.GaussianBlur(change_mask, (11, 11), 0)
            debug_images["Change Mask (Blurred)"] = change_mask_blurred

            # Detect ellipses from the change mask
            try:
                # Morph close to connect regions for more stable contours
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                cleaned_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel_close)

                mask_ellipses, mask_areas = _extract_ellipse_candidates_from_binary(
                    cleaned_mask,
                    min_area=min_area_mask,
                    min_axis=min_axis_len,
                    max_aspect=8.0,
                )
            except Exception as e:
                print(f"Failed to detect ellipses from change mask: {e}")
                mask_ellipses = []
                mask_areas = []
        except Exception as e:
            print(f"Failed to generate change mask: {e}")
            mask_ellipses = []
            mask_areas = []
    else:
        mask_ellipses = []
        mask_areas = []

    # Store FAST detection parameters for consistency in feature extraction
    FAST_THRESHOLD = 15
    FAST_MIN_CIRCULARITY = 0.4
    FAST_SIZE_TOLERANCE = 5
    
    fast_keypoints, keypoint_sizes = detect_fast_circles(gray, threshold=FAST_THRESHOLD, min_circularity=FAST_MIN_CIRCULARITY)
    
    # Keep all FAST keypoints for scoring, and keep mode-size points for the
    # final best-ellipse preview.
    mode_size = find_mode_size(keypoint_sizes, tolerance=FAST_SIZE_TOLERANCE)
    mode_keypoints = (
        filter_keypoints_by_size(
            fast_keypoints,
            keypoint_sizes,
            mode_size,
            tolerance=FAST_SIZE_TOLERANCE,
        )
        if mode_size is not None
        else fast_keypoints
    )

    print(f"Found {len(fast_keypoints)} FAST keypoints")

    # Draw all FAST keypoints on debug output
    cv2.drawKeypoints(debug_output, fast_keypoints, debug_output, color=(0, 255, 255),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    debug_images["FAST Keypoints"] = debug_output

    # Blurred image ONLY for Portafilter Ellipse Detection (not for feature detection)
    el_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    debug_images["Blurred"] = el_gray

    # Edge detection
    # Use original/sharpened image for Hough line detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Use blurred image ONLY for ellipse detection
    el_edges = cv2.Canny(el_gray, 90, 180, apertureSize=3)
    debug_images["Canny Edges"] = edges

    # Additional edge maps improve ellipse candidate recall under varying lighting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    el_edges_alt = cv2.Canny(clahe, 70, 160, apertureSize=3)
    el_edges_light = cv2.Canny(gray, 60, 150, apertureSize=3)

    candidate_edge_map = cv2.bitwise_or(el_edges, el_edges_alt)
    candidate_edge_map = cv2.bitwise_or(candidate_edge_map, el_edges_light)
    candidate_edge_map = cv2.morphologyEx(
        candidate_edge_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    debug_images["Ellipse Candidate Edges"] = candidate_edge_map

    #  1. Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=20, maxLineGap=10)
    max_segment_length = 50  # Maximum length per segment in pixels
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Compute total line length and direction
            dx = x2 - x1
            dy = y2 - y1
            length = np.hypot(dx, dy)

            if length <= max_segment_length:
                # Line is short enough — draw as is
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 1)
            else:
                # Break line into smaller segments
                num_segments = int(np.ceil(length / max_segment_length))
                for i in range(num_segments):
                    t1 = i / num_segments
                    t2 = (i + 1) / num_segments
                    sx1 = int(x1 + t1 * dx)
                    sy1 = int(y1 + t1 * dy)
                    sx2 = int(x1 + t2 * dx)
                    sy2 = int(y1 + t2 * dy)
                    cv2.line(output, (sx1, sy1), (sx2, sy2), (0, 255, 0), 1)

    # Draw filtered keypoints on output
    for kp in fast_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output, (x, y), 3, (255, 0, 0), -1)  # Draw small circles

    ellipses = []
    areas = []
    if manual_ellipse is None:
        if yolo_bbox is not None:
            forced_bbox_ellipse = _ellipse_from_roi_bounds(
                image.shape,
                pad_ratio_x=0.08,
                pad_ratio_y=0.10,
            )
            if forced_bbox_ellipse is not None:
                forced_area = float(forced_bbox_ellipse[1][0] * forced_bbox_ellipse[1][1] * np.pi)
                if _append_unique_ellipse(ellipses, areas, forced_bbox_ellipse, forced_area):
                    safe_draw_ellipse(output, forced_bbox_ellipse, (255, 255, 0), 2)
                    print("Added forced bbox-derived ellipse candidate.")

        # 3. Ellipse detection from a combined edge map for better recall.
        edge_ellipses, edge_areas = _extract_ellipse_candidates_from_binary(
            candidate_edge_map,
            min_area=min_area_auto,
            min_axis=min_axis_len,
            max_aspect=8.0,
        )
        for ellipse, area in zip(edge_ellipses, edge_areas):
            if _append_unique_ellipse(ellipses, areas, ellipse, area):
                safe_draw_ellipse(output, ellipse, (0, 0, 255), 1)

        # Merge in mask-derived ellipses
        for m_el, m_area in zip(mask_ellipses, mask_areas):
            if _append_unique_ellipse(ellipses, areas, m_el, m_area):
                safe_draw_ellipse(output, m_el, (0, 255, 0), 1)
    else:
        safe_draw_ellipse(output, manual_ellipse, (0, 255, 255), 2)
        debug_images["Manual Ellipse"] = output.copy()

    # Step 4: Score and highlight best ellipse with FAST features
    if manual_ellipse is not None:
        best_ellipse = manual_ellipse
        print("Using manually selected ellipse.")
    else:
        print("Evaluating ellipses with FAST features")
        best_ellipse = ellipse_feature_score(
            image,
            ellipses,
            areas,
            fast_keypoints=fast_keypoints,
            min_area=500,
        )
        if best_ellipse is None and len(ellipses) > 0:
            print("WARNING: No high-score ellipse found - using largest fallback ellipse.")
            best_ellipse = max(
                ellipses,
                key=lambda el: el[1][0] * el[1][1]  # selects by width × height
            )

    detected_features_view = output.copy()
    if best_ellipse is not None:
        safe_draw_ellipse(detected_features_view, best_ellipse, (255, 0, 255), 4)
        (sel_cx, sel_cy), _, _ = best_ellipse
        cv2.putText(
            detected_features_view,
            "Selected",
            (int(sel_cx) - 35, max(18, int(sel_cy) - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    debug_images["Detected Features"] = detected_features_view
    
    # Create visualization showing FAST keypoints within the best ellipse
    best_ellipse_keypoints = []
    if best_ellipse is not None:
        best_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.ellipse(best_mask, best_ellipse, 255, -1)

        # Final overlay uses mode-sized points inside the chosen ellipse.
        final_keypoints = mode_keypoints if mode_keypoints else fast_keypoints
        for kp in final_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < image.shape[0] and 0 <= x < image.shape[1] and best_mask[y, x] == 255:
                best_ellipse_keypoints.append(kp)

    if best_ellipse is not None and len(best_ellipse_keypoints) > 0:

        # Keep the final preview focused: only the final ellipse and interior keypoints.
        cv2.drawKeypoints(output2, best_ellipse_keypoints, output2, color=(255, 0, 255), 
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        safe_draw_ellipse(output2, best_ellipse, (255, 0, 255), 15)

        (cx, cy), (major, minor), angle = best_ellipse
        cv2.putText(output2, f"FAST: {len(best_ellipse_keypoints)}", 
                    (int(cx-50), int(cy-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print(f"Best ellipse contains {len(best_ellipse_keypoints)} FAST keypoints")

    if best_ellipse is not None and len(best_ellipse_keypoints) == 0:
        # Fallback view when keypoints are not available: show only the selected final ellipse.
        safe_draw_ellipse(output2, best_ellipse, (255, 0, 255), 15)
    
    # Show comparison only when dashboard/debug visualisation is requested.
    if save_dashboard and best_ellipse is not None and len(best_ellipse_keypoints) > 0:
        def is_same_ellipse(e1, e2, tol=1e-1):
            if e1 is None or e2 is None:
                return False
            return (
                abs(e1[0][0] - e2[0][0]) < tol and
                abs(e1[0][1] - e2[0][1]) < tol and
                abs(e1[1][0] - e2[1][0]) < tol and
                abs(e1[1][1] - e2[1][1]) < tol
            )

        comparison_img = image.copy()
        for ellipse in ellipses:
            is_mask = any(is_same_ellipse(ellipse, m_el) for m_el in mask_ellipses)
            is_best = is_same_ellipse(ellipse, best_ellipse)
            if is_mask:
                color, thickness = (0, 255, 0), 3
            elif is_best:
                color, thickness = (0, 255, 255), 3
            else:
                color, thickness = (0, 0, 255), 1
            safe_draw_ellipse(comparison_img, ellipse, color, thickness)

            # Count FAST points in this ellipse once per ellipse.
            ellipse_count = count_keypoints_in_ellipse(ellipse, mode_keypoints, image.shape)
            (cx, cy), _, _ = ellipse
            cv2.putText(
                comparison_img,
                f"{ellipse_count}",
                (int(cx - 20), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        debug_images["Ellipse Comparison"] = comparison_img

    best_ellipse_full = _offset_ellipse(best_ellipse, roi_offset)

    if yolo_bbox is not None:
        x1, y1, x2, y2 = yolo_bbox
        full_result = original_image.copy()
        target_h = max(1, y2 - y1)
        target_w = max(1, x2 - x1)
        patch = output2
        if patch.shape[0] != target_h or patch.shape[1] != target_w:
            patch = cv2.resize(patch, (target_w, target_h))
        full_result[y1:y2, x1:x2] = patch
        cv2.rectangle(full_result, (x1, y1), (x2, y2), (0, 255, 255), 2)
        output2 = full_result

    debug_images["Final Result"] = output2
    
    # Use original image for final cropping (not sharpened or blurred)
    cropped_img = crop_image_by_ellipse(original_image, best_ellipse_full)
    debug_images["Cropped Result"] = cropped_img

    # Reorder debug images for better slideshow flow.
    ordered_debug_images = {}
    
    # Add images in desired order
    if "Original" in debug_images:
        ordered_debug_images["Original"] = debug_images["Original"]
    if "YOLO ROI" in debug_images:
        ordered_debug_images["YOLO ROI"] = debug_images["YOLO ROI"]
    if "Sharpened" in debug_images:
        ordered_debug_images["Sharpened"] = debug_images["Sharpened"]
    if "Blurred" in debug_images:
        ordered_debug_images["Blurred"] = debug_images["Blurred"]
    if "Canny Edges" in debug_images:
        ordered_debug_images["Canny Edges"] = debug_images["Canny Edges"]
    if "Change Mask (Blurred)" in debug_images:
        ordered_debug_images["Change Mask (Blurred)"] = debug_images["Change Mask (Blurred)"]
    if "FAST Keypoints" in debug_images:
        ordered_debug_images["FAST Keypoints"] = debug_images["FAST Keypoints"]
    if "Detected Features" in debug_images:
        ordered_debug_images["Detected Features"] = debug_images["Detected Features"]
    if "Manual Ellipse" in debug_images:
        ordered_debug_images["Manual Ellipse"] = debug_images["Manual Ellipse"]
    if "Ellipse Comparison" in debug_images:
        ordered_debug_images["Ellipse Comparison"] = debug_images["Ellipse Comparison"]
    if "Final Result" in debug_images:
        ordered_debug_images["Final Result"] = debug_images["Final Result"]
    if "Cropped Result" in debug_images:
        ordered_debug_images["Cropped Result"] = debug_images["Cropped Result"]
    # Only show dashboards if saving or explicitly requested
    if save_dashboard:
        if use_interactive:
            create_interactive_dashboard(ordered_debug_images)
        else:
            create_debug_dashboard(ordered_debug_images, save_path=dashboard_path)

    # Return FAST detection parameters for consistency in feature extraction.
    fast_params = {
        'threshold': FAST_THRESHOLD,
        'min_circularity': FAST_MIN_CIRCULARITY,
        'size_tolerance': FAST_SIZE_TOLERANCE,
        'mode_size': mode_size
    }
    
    if return_debug:
        return output2, best_ellipse_full, mode_size, fast_params, ordered_debug_images
    return output2, best_ellipse_full, mode_size, fast_params

# Test Run
if __name__ == "__main__":
    frames_dir_guess = os.path.join(Base_Dir, "Image Data", "Frames")
    pf_dir_guess = Frame_Dir

    # Try to load first and 20th frames
    candidates_first = [
        os.path.join(frames_dir_guess, "frame_0001.jpg"),
        os.path.join(pf_dir_guess, "frame_0001.jpg"),
    ]
    candidates_twentieth = [
        os.path.join(frames_dir_guess, "frame_0020.jpg"),
        os.path.join(pf_dir_guess, "frame_0020.jpg"),
    ]

    img_path = next((p for p in candidates_first if os.path.exists(p)), None)
    second_path = next((p for p in candidates_twentieth if os.path.exists(p)), None)

    if img_path is None:
        print("First frame not found in expected locations.")
        print("Checked:")
    
    if img_path is not None:
        frame = load_image_with_orientation(img_path)
        second_frame = load_image_with_orientation(second_path) if second_path else None
        if second_frame is None:
            print("20th frame not found; running without change mask.")
        else:
            print("Using change mask between first and 20th frames.")

        use_interactive = True

        _, result, mode_size, fast_params = detect_elliptical_portafilter_with_holes(
            frame,
            save_dashboard=True,
            dashboard_path="portafilter_detection_dashboard.png",
            use_interactive=use_interactive,
            second_frame=second_frame,
            mask_threshold=15,
        )

        if result is not None:
            print("Detection completed successfully!")
            if mode_size is not None:
                print(f"Detected hole mode size: {mode_size}")
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            print("Failed to detect portafilter")
