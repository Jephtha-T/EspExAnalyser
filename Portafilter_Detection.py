import cv2
import os
import numpy as np
from PIL import Image, ExifTags
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import matplotlib.patches as mpatches

Base_Dir = os.path.dirname(os.path.abspath(__file__))
Frame_Dir = os.path.join(Base_Dir, "Image Data/Frames")

def create_interactive_dashboard(images_dict, save_path=None):
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
        """Update the current slide"""
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
        """Go to next slide"""
        nonlocal current_index
        current_index = (current_index + 1) % len(images)
        update_slide()
    
    def prev_slide(event):
        """Go to previous slide"""
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

def detect_surf_corners(image, hessian_threshold=400, min_circularity=0.7):
    # Try to use SURF if available (patented, may not be available in all OpenCV builds)
    try:
        xfeatures2d = getattr(cv2, 'xfeatures2d', None)
        if xfeatures2d is not None:
            surf_create = getattr(xfeatures2d, 'SURF_create', None)
            if surf_create is not None:
                surf = surf_create(hessian_threshold)
                keypoints = surf.detect(image, None)
                print("Using SURF detector")
            else:
                raise AttributeError('SURF_create not available')
        else:
            raise AttributeError('xfeatures2d not available')
    except Exception as e:
        print(f"SURF not available ({e}), using FAST corner detector as fallback")
        # Use getattr to avoid linter errors
        fast_create = getattr(cv2, 'FastFeatureDetector_create', None)
        if fast_create is not None:
            fast = fast_create(threshold=15)  # Lower threshold for more sensitivity
        else:
            fast_class = getattr(cv2, 'FastFeatureDetector', None)
            if fast_class is not None:
                fast = fast_class.create(threshold=15)  # Lower threshold for more sensitivity
            else:
                raise RuntimeError('No FAST detector available in this OpenCV build')
        keypoints = fast.detect(image, None)
        print("Using FAST detector")
    
    # Filter keypoints for circularity
    if keypoints:
        filtered_keypoints = []
        height, width = image.shape[:2]
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Handle different keypoint types (SURF vs FAST)
            if hasattr(kp, 'size'):
                size = int(kp.size)
            else:
                # FAST keypoints don't have size, use a reasonable default
                size = 20
            
            # Check if keypoint is within image bounds
            if x < size//2 or y < size//2 or x >= width - size//2 or y >= height - size//2:
                continue
            
            # Extract region around keypoint
            x1, y1 = max(0, int(x - size//2)), max(0, int(y - size//2))
            x2, y2 = min(width, int(x + size//2)), min(height, int(y + size//2))
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # Calculate circularity using contour analysis
            try:
                # Create binary mask from region
                _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > 0:
                        # Calculate circularity
                        perimeter = cv2.arcLength(largest_contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            # Only keep keypoints with good circularity
                            if circularity >= min_circularity:
                                filtered_keypoints.append(kp)
            except Exception as e:
                # If circularity calculation fails, keep the keypoint
                print(f"Circularity calculation failed for keypoint at ({x}, {y}): {e}")
                filtered_keypoints.append(kp)
        
        #print(f"Filtered {len(keypoints)} keypoints to {len(filtered_keypoints)} circular features")
        return filtered_keypoints
    
    return keypoints

def filter_surf_keypoints_by_ellipse(keypoints, ellipse, image_shape):
    if ellipse is None or keypoints is None:
        return []
    
    (cx, cy), (major, minor), angle = ellipse
    height, width = image_shape[:2]
    
    # Create mask for the ellipse
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
    
    filtered_keypoints = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        if 0 <= x < width and 0 <= y < height and mask[y, x] == 255:
            filtered_keypoints.append(kp)
    
    return filtered_keypoints

def safe_draw_ellipse(img, ellipse, color, thickness=1):
    if ellipse is None:
        return
    if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
        return
    if not (np.isfinite(ellipse[1][0]) and np.isfinite(ellipse[1][1])):
        return
    cv2.ellipse(img, ellipse, color, thickness)

def ellipse_feature_score(image, lines, circles_list, ellipses, areas, surf_keypoints=None, fast_keypoints=None,
                        MIN_FEATURES=5, MIN_AREA=500, MAX_AREA=50000,
                        CENTER_BIAS_WEIGHT=0.02, SURF_WEIGHT=5.0, EMPTY_PENALTY_WEIGHT=50.0):
    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    best_ellipse = None
    best_score = float('-inf')
    n=0
    print(len(ellipses), "ellipses found")

    for el in ellipses:
        #print(f"Evaluating ellipse: {el}")
        # Basic ellipse info
        (cx, cy), (x_len, y_len), angle = el
        if not (np.isfinite(x_len) and np.isfinite(y_len)):
            #print(f"Skipping invalid ellipse: {el} infinite dimensions")
            continue
        if x_len <= 0 or y_len <= 0:
            #print(f"Skipping invalid ellipse: {el} non-positive dimensions")
            continue

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, el, (255, 255, 255), -1)

        area = np.count_nonzero(mask)
        area = areas[n] if n < len(areas) else area  # Use precomputed area
        n += 1
        if area < MIN_AREA:
            #print(f"Skipping ellipse: {el} due to small area ({area})")
            continue

        # Count weighted features
        feature_score = 0
        surf_score = 0

        # Count Hough lines within ellipse
        if lines is not None:
            for line in lines:
                for x, y in [(line[0][0], line[0][1]), (line[0][2], line[0][3])]:
                    if y < height and x < width and mask[y, x]:
                        feature_score += 1

        # Count FAST circles within ellipse
        if fast_keypoints is not None:
            for kp in fast_keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < width and 0 <= y < height and mask[y, x]:
                    feature_score += 2  # FAST circles get higher weight as they're more distinctive

        # Calculate overall feature density score (all features per ellipse area)
        density_score = calculate_feature_density_score(el, image.shape, surf_keypoints, lines, fast_keypoints)
        
        # Add center-weighted score for SURF keypoints if available
        center_weighted_score = 0
        if surf_keypoints is not None:
            filtered_keypoints = filter_surf_keypoints_by_ellipse(surf_keypoints, el, image.shape)
            center_weighted_score = calculate_center_weighted_score(filtered_keypoints, el, image.shape, center_weight=2.0)
            
            #print(f"Ellipse {el}: Found {len(filtered_keypoints)} SURF keypoints")
            #print(f"  - Center-weighted score: {center_weighted_score:.2f}")
        
        # Combine scores: density is the primary factor
        feature_score = density_score * 10000 + center_weighted_score * SURF_WEIGHT  # Scale density up for better comparison
        
        #print(f"  - Feature density score: {density_score:.6f}")
        #print(f"  - Final feature score: {feature_score:.2f}")

        # Require minimum SURF features for a valid portafilter
        if surf_keypoints is not None and len(filtered_keypoints) < 3:  # At least 3 SURF points
            #print(f"Skipping ellipse: {el} due to insufficient SURF features ({len(filtered_keypoints)})")
            continue

        if feature_score < 5:
            #print(f"Skipping ellipse: {el} due to insufficient total features ({feature_score})")
            continue

        # Penalize off-center ellipses
        dist_from_center = np.linalg.norm(np.array([cx, cy]) - image_center)
        center_penalty = dist_from_center * CENTER_BIAS_WEIGHT

        # Final score: density-based with center bias
        score = feature_score - center_penalty
        #print(f"Ellipse: {el}, Score: {score:.2f}, Area: {area}, Features: {feature_score:.2f}")

        if score > best_score:
            best_score = score
            best_ellipse = el
    print(f"Best Ellipse: {best_ellipse}, Score: {best_score:.2f}")
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



def calculate_center_weighted_score(surf_keypoints, ellipse, image_shape, center_weight=2.0):
    if not surf_keypoints or ellipse is None:
        return 0
    
    (cx, cy), (major, minor), angle = ellipse
    height, width = image_shape[:2]
    image_center = np.array([width // 2, height // 2])
    
    total_score = 0
    for kp in surf_keypoints:
        x, y = kp.pt[0], kp.pt[1]
        keypoint_pos = np.array([x, y])
        
        # Calculate distance from image center
        dist_from_center = np.linalg.norm(keypoint_pos - image_center)
        max_dist = np.linalg.norm(np.array([width, height]))
        
        # Normalize distance (0 = center, 1 = corner)
        normalized_dist = dist_from_center / max_dist
        
        # Weight: closer to center = higher weight
        weight = 1 + (center_weight - 1) * (1 - normalized_dist)
        
        total_score += weight
    
    return total_score



def calculate_feature_density_score(ellipse, image_shape, surf_keypoints, lines, fast_keypoints):
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
    
    # Count features within ellipse
    feature_count = 0
    
    # Count SURF keypoints within ellipse
    if surf_keypoints is not None:
        filtered_surf = filter_surf_keypoints_by_ellipse(surf_keypoints, ellipse, image_shape)
        feature_count += len(filtered_surf)
    
    # Count Hough lines within ellipse
    if lines is not None:
        line_count = 0
        for line in lines:
            for x, y in [(line[0][0], line[0][1]), (line[0][2], line[0][3])]:
                if 0 <= y < height and 0 <= x < width and mask[y, x]:
                    line_count += 1
        feature_count += line_count
    
    # Count FAST circles within ellipse
    if fast_keypoints is not None:
        circle_count = 0
        for kp in fast_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= x < width and 0 <= y < height and mask[y, x]:
                circle_count += 1
        feature_count += circle_count
    
    # Calculate density (features per pixel)
    density = (feature_count**2) / (ellipse_area/10)

    #print(f"  Density Analysis:")
    #print(f"    - Ellipse area: {ellipse_area} pixels")
    #print(f"    - Total features: {feature_count}")
    #print(f"    - Feature density: {density:.6f} features/pixel")
    
    return density

def detect_fast_circles(image, threshold=25, min_circularity=0.6):
    # Use FAST corner detector
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
        print("Using FAST detector for circle detection")
    except Exception as e:
        print(f"FAST detection failed: {e}")
        return []
    
    # Filter keypoints for circularity and collect sizes
    if keypoints:
        filtered_keypoints = []
        keypoint_sizes = []
        height, width = image.shape[:2]
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Handle different keypoint types (SURF vs FAST)
            if hasattr(kp, 'size'):
                size = int(kp.size)
            else:
                # FAST keypoints don't have size, use a reasonable default
                size = 20
            
            # Check if keypoint is within image bounds
            if x < size//2 or y < size//2 or x >= width - size//2 or y >= height - size//2:
                continue
            
            # Extract region around keypoint
            x1, y1 = max(0, int(x - size//2)), max(0, int(y - size//2))
            x2, y2 = min(width, int(x + size//2)), min(height, int(y + size//2))
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                continue
            
            # Calculate circularity using contour analysis
            try:
                # Create binary mask from region
                _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    
                    if area > 0:
                        # Calculate circularity
                        perimeter = cv2.arcLength(largest_contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            
                            # Only keep keypoints with good circularity
                            if circularity >= min_circularity:
                                filtered_keypoints.append(kp)
                                keypoint_sizes.append(size)
            except Exception as e:
                # If circularity calculation fails, keep the keypoint
                #print(f"Circularity calculation failed for keypoint at ({x}, {y}): {e}")
                filtered_keypoints.append(kp)
                keypoint_sizes.append(size)
        
        #print(f"Filtered {len(keypoints)} keypoints to {len(filtered_keypoints)} circular features")
        return filtered_keypoints, keypoint_sizes
    
    return [], []

def find_mode_size(keypoint_sizes, tolerance=5):
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
        #print(f"Mode size: {mode_size} (appears {size_groups[mode_size]} times)")
        return mode_size
    
    return None

def filter_keypoints_by_size(keypoints, keypoint_sizes, target_size, tolerance=5):
    if target_size is None:
        return keypoints
    
    filtered_keypoints = []
    for kp, size in zip(keypoints, keypoint_sizes):
        if abs(size - target_size) <= tolerance:
            filtered_keypoints.append(kp)
    
    #print(f"Filtered to {len(filtered_keypoints)} keypoints with mode size {target_size}")
    return filtered_keypoints

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
):
    if image is None:
        print(f"Error loading image: {image}")
        return None
    
    # Dictionary to store all intermediary steps for dashboard
    debug_images = {}
    debug_images["Original"] = image.copy()
    
    output = image.copy()
    output2 = image.copy()
    debug_output = image.copy()  # For showing SURF evaluation process

    # Check if image is too sharp or too blurry
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(f"Laplacian variance: {lap_var:.2f}")
    
    # Store original image for final cropping
    original_image = image.copy()
    
    if lap_var > 1000:
        print(f"WARNING: Image is very sharp (Laplacian variance: {lap_var:.2f}) - blurring with 7")
        image = cv2.medianBlur(image, 7)
        debug_images["Blurred"] = image.copy()
    elif lap_var > 500:
        print(f"WARNING: Image is too sharp (Laplacian variance: {lap_var:.2f}) - blurring with 5")
        image = cv2.medianBlur(image, 5)
        debug_images["Blurred"] = image.copy()
    elif lap_var < 50:
        print(f"WARNING: Image is very blurry (Laplacian variance: {lap_var:.2f}) - sharpening")
        # Sharpening kernel
        sharpen_kernel = np.array([[0, -3, 0],
                                [-3, 15, -3],
                                [0, -3, 0]], dtype=np.float32)
        image = cv2.filter2D(image, -1, sharpen_kernel)
        debug_images["Sharpened"] = image.copy()
    elif lap_var < 100:
        print(f"WARNING: Image is too blurry (Laplacian variance: {lap_var:.2f}) - sharpening")
        # Sharpening kernel
        sharpen_kernel = np.array([[0, -2, 0],
                                [-2, 9, -2],
                                [0, -2, 0]], dtype=np.float32)
        image = cv2.filter2D(image, -1, sharpen_kernel)
        debug_images["Sharpened"] = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #  Change Mask between first frame (image) and provided second_frame 
    if second_frame is not None:
        try:
            change_mask = generate_change_mask(image, second_frame, threshold=mask_threshold)
            change_mask_blurred = cv2.GaussianBlur(change_mask, (11, 11), 0)
            debug_images["Change Mask (Blurred)"] = change_mask_blurred

            # Detect ellipses from the change mask
            mask_ellipses = []
            mask_areas = []
            try:
                # Morph close to connect regions for more stable contours
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                cleaned_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel_close)

                contours_mask, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_mask:
                    if len(cnt) < 5:
                        continue
                    ellipse = cv2.fitEllipse(cnt)
                    (x_len, y_len) = ellipse[1]
                    # Basic validation
                    if x_len <= 0 or y_len <= 0:
                        continue
                    aspect_ratio = max(x_len, y_len) / max(1e-6, min(x_len, y_len))
                    if aspect_ratio > 10 or min(x_len, y_len) < 10:
                        continue
                    area = x_len * y_len * np.pi
                    mask_ellipses.append(ellipse)
                    mask_areas.append(area)
            except Exception as e:
                print(f"Failed to detect ellipses from change mask: {e}")
        except Exception as e:
            print(f"Failed to generate change mask: {e}")
    else:
        mask_ellipses = []
        mask_areas = []

    #  FAST Corner Detection for Portafilter Holes
    # Use original/sharpened image for optimal feature detection (not blurred)
    print("Detecting FAST corners for portafilter holes...")
    surf_keypoints = detect_surf_corners(gray, hessian_threshold=250, min_circularity=0.5)
    print(f"Found {len(surf_keypoints)} FAST keypoints")

    # Draw all FAST keypoints on debug output
    cv2.drawKeypoints(debug_output, surf_keypoints, debug_output, color=(0, 255, 255),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    debug_images["FAST Keypoints"] = debug_output

    # Blurred image ONLY for Portafilter Ellipse Detection (not for feature detection)
    el_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    debug_images["Blurred"] = el_gray

    # Edge detection
    # Use original/sharpened image for Hough line detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # Use blurred image ONLY for ellipse detection
    el_edges = cv2.Canny(el_gray, 100, 200, apertureSize=3)
    debug_images["Canny Edges"] = edges

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

    # 2. FAST Circle Detection
    # Use original/sharpened image for optimal circle detection (not blurred)
    print("Detecting circles using FAST...")
    fast_keypoints, keypoint_sizes = detect_fast_circles(gray, threshold=15, min_circularity=0.4)
    
    # Find mode size of circles
    mode_size = find_mode_size(keypoint_sizes, tolerance=5)
    
    # Filter keypoints to only use mode-sized circles
    if mode_size is not None:
        filtered_keypoints = filter_keypoints_by_size(fast_keypoints, keypoint_sizes, mode_size, tolerance=5)
    else:
        filtered_keypoints = fast_keypoints

    # Draw filtered keypoints on output
    for kp in filtered_keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(output, (x, y), 3, (255, 0, 0), -1)  # Draw small circles

    # 3. Ellipse Detection from Edges
    contours, _ = cv2.findContours(el_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipses = []
    areas = []
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            area = ellipse[1][0] * ellipse[1][1] * np.pi # Calculate the area of the fitted ellipse
            if area < 50000 :
                continue
            if ellipse[1][0] <= 0 or ellipse[1][1] <= 0:
                continue  # skip bad ellipse
            (x_len, y_len) = ellipse[1]
            if x_len <= 0 or y_len <= 0:
                continue

            # Reject too flat (line-like) ellipses
            aspect_ratio = max(x_len, y_len) / min(x_len, y_len)
            if aspect_ratio > 10 or min(x_len, y_len) < 10:
                continue

            ellipses.append(ellipse)
            areas.append(area)
            safe_draw_ellipse(output, ellipse, (0, 0, 255), 1)

    # Merge in mask-derived ellipses
    for m_el, m_area in zip(mask_ellipses, mask_areas):
        ellipses.append(m_el)
        areas.append(m_area)
        safe_draw_ellipse(output, m_el, (0, 255, 0), 1)

    debug_images["Detected Features"] = output

    # Step 4: Score and highlight best ellipse with SURF features
    print("Evaluating ellipses with SURF features")
    best_ellipse = ellipse_feature_score(image, lines, None, ellipses, areas, surf_keypoints, fast_keypoints)
    if best_ellipse is None and len(ellipses) > 0:
        print("WARNING: No high-score ellipse found - using largest fallback ellipse.")
        best_ellipse = max(
            ellipses,
            key=lambda el: el[1][0] * el[1][1]  # selects by width × height
        )
    
    # Create visualization showing SURF keypoints within the best ellipse
    if best_ellipse is not None and surf_keypoints is not None:
        filtered_keypoints = filter_surf_keypoints_by_ellipse(surf_keypoints, best_ellipse, image.shape)
        
        # Draw the best ellipse and its SURF keypoints
        cv2.drawKeypoints(output2, filtered_keypoints, output2, color=(255, 0, 255), 
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # If best ellipse originates from mask set, draw it green; otherwise yellow
        def is_same_ellipse(e1, e2, tol=1e-1):
            if e1 is None or e2 is None:
                return False
            return (
                abs(e1[0][0] - e2[0][0]) < tol and
                abs(e1[0][1] - e2[0][1]) < tol and
                abs(e1[1][0] - e2[1][0]) < tol and
                abs(e1[1][1] - e2[1][1]) < tol
            )

        best_is_mask = any(is_same_ellipse(best_ellipse, m_el) for m_el in mask_ellipses)
        if best_is_mask:
            safe_draw_ellipse(output2, best_ellipse, (0, 255, 0), 15)  # green for mask-derived
        else:
            safe_draw_ellipse(output2, best_ellipse, (255, 0, 255), 15)  # red otherwise
        
        # Add text showing SURF count
        (cx, cy), (major, minor), angle = best_ellipse
        cv2.putText(output2, f"SURF: {len(filtered_keypoints)}", 
                    (int(cx-50), int(cy-50)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        print(f"Best ellipse contains {len(filtered_keypoints)} SURF keypoints")
        
        # Show comparison: all ellipses with their SURF counts
        comparison_img = image.copy()
        for i, ellipse in enumerate(ellipses):
            # Determine color: green for mask-derived, yellow for best, red for others
            is_mask = any(is_same_ellipse(ellipse, m_el) for m_el in mask_ellipses)
            is_best = is_same_ellipse(ellipse, best_ellipse)
            if is_mask:
                color, thickness = (0, 255, 0), 3
            elif is_best:
                color, thickness = (0, 255, 255), 3
            else:
                color, thickness = (0, 0, 255), 1
            
            safe_draw_ellipse(comparison_img, ellipse, color, thickness)
            
            # Count SURF points in this ellipse
            if surf_keypoints is not None:
                ellipse_keypoints = filter_surf_keypoints_by_ellipse(surf_keypoints, ellipse, image.shape)
                (cx, cy), _, _ = ellipse
                cv2.putText(comparison_img, f"{len(ellipse_keypoints)}", 
                            (int(cx-20), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        debug_images["Ellipse Comparison"] = comparison_img
    
    # Also overlay all mask-derived ellipses in green on the final result for clarity
    for m_el in mask_ellipses:
        safe_draw_ellipse(output2, m_el, (0, 255, 0), 4)
    debug_images["Final Result"] = output2
    
    # Use original image for final cropping (not sharpened or blurred)
    cropped_img = crop_image_by_ellipse(original_image, best_ellipse)
    debug_images["Cropped Result"] = cropped_img

    # Reorder debug images for better slideshow flow
    ordered_debug_images = {}
    
    # Add images in desired order
    if "Original" in debug_images:
        ordered_debug_images["Original"] = debug_images["Original"]
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
    if "Ellipse Comparison" in debug_images:
        ordered_debug_images["Ellipse Comparison"] = debug_images["Ellipse Comparison"]
    if "Final Result" in debug_images:
        ordered_debug_images["Final Result"] = debug_images["Final Result"]
    if "Cropped Result" in debug_images:
        ordered_debug_images["Cropped Result"] = debug_images["Cropped Result"]
    # Only show dashboards if saving or explicitly requested
    if save_dashboard:
        if use_interactive:
            create_interactive_dashboard(ordered_debug_images, save_path=dashboard_path)
        else:
            create_debug_dashboard(ordered_debug_images, save_path=dashboard_path)


    return output2, best_ellipse

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

        _, result = detect_elliptical_portafilter_with_holes(
            frame,
            save_dashboard=True,
            dashboard_path="portafilter_detection_dashboard.png",
            use_interactive=use_interactive,
            second_frame=second_frame,
            mask_threshold=15,
        )

        if result is not None:
            print("Detection completed successfully!")
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            print("Failed to detect portafilter")