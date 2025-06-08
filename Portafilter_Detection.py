import cv2
import os
import numpy as np
from PIL import Image, ExifTags

base_dir = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(base_dir, "Image Data/Frames")

def load_image_with_orientation(path):
    image = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
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

def ellipse_feature_score(image, lines, circles, ellipses, areas,
                        MIN_FEATURES=5, MIN_AREA=500, MAX_AREA=50000,
                        CENTER_BIAS_WEIGHT=0.02):
    height, width = image.shape[:2]
    image_center = np.array([width // 2, height // 2])
    best_ellipse = None
    best_score = float('-inf')
    n=0
    print(len(ellipses), "ellipses found")

    for el in ellipses:
        print(f"Evaluating ellipse: {el}")
        # Basic ellipse info
        (cx, cy), (x_len, y_len), angle = el
        if not (np.isfinite(x_len) and np.isfinite(y_len)):
            print(f"Skipping invalid ellipse: {el} infinite dimensions")
            continue
        if x_len <= 0 or y_len <= 0:
            print(f"Skipping invalid ellipse: {el} non-positive dimensions")
            continue

        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, el, 255, -1)

        area = np.count_nonzero(mask)
        area = areas[n] if n < len(areas) else area  # Use precomputed area
        n += 1
        if area < MIN_AREA:
            print(f"Skipping ellipse: {el} due to small area ({area})")
            continue

        # Count weighted features
        feature_score = 0

        if lines is not None:
            for line in lines:
                for x, y in [(line[0][0], line[0][1]), (line[0][2], line[0][3])]:
                    if y < height and x < width and mask[y, x]:
                        feature_score += 1
            # for line in lines:
            #     x1, y1, x2, y2 = line[0]
            #     for alpha in np.linspace(0, 1, 10):  # 10 samples along the line
            #         x = int(x1 * (1 - alpha) + x2 * alpha)
            #         y = int(y1 * (1 - alpha) + y2 * alpha)
            #         if 0 <= x < width and 0 <= y < height and mask[y, x]:
            #             dist = np.linalg.norm(np.array([x, y]) - image_center)
            #             weight = 1 / (1 + dist)
            #             feature_score += weight


        # if circles is not None:
        #     for x, y, r in circles:
        #         if int(y) < height and int(x) < width and mask[int(y), int(x)]:
        #             dist = np.linalg.norm(np.array([x, y]) - image_center)
        #             weight = 1 / (1 + dist)
        #             feature_score += weight

        if feature_score < 5:
            print(f"Skipping ellipse: {el} due to insufficient features ({feature_score})")
            continue

        # Penalize off-center ellipses
        dist_from_center = np.linalg.norm(np.array([cx, cy]) - image_center)
        center_penalty = dist_from_center * CENTER_BIAS_WEIGHT

        empty_penalty = (area - feature_score * 100)  # scale up feature density

        # Final score = dense features / area - penalty
        # score = (feature_score ** 1.5) / (area + 1e-5) - center_penalty
        #score = ((feature_score ** 1.5) / (area + 1e-5)) - center_penalty - 0.0005 * empty_penalty
        score = (feature_score ** 2) / ((area / 10)) - center_penalty
        print(f"Ellipse: {el}, Score: {score:.2f}, Area: {area}, Features: {feature_score:.2f}")


        if score > best_score:
            best_score = score
            best_ellipse = el
    print(f"Best Ellipse: {best_ellipse}, Score: {best_score:.2f}")
    return best_ellipse

def crop_image_by_ellipse(image, ellipse, padding=10):
    (cx, cy), (major, minor), angle = ellipse

    theta_rad = np.deg2rad(angle)

    # Major axis vector (horizontal extent)
    dx_major = (major / 2) * np.cos(theta_rad)
    dy_major = (major / 2) * np.sin(theta_rad)

    # Minor axis vector (vertical extent)
    dx_minor = (minor / 2) * np.sin(theta_rad)
    dy_minor = -(minor / 2) * np.cos(theta_rad)

    # Compute points
    left_x = int(cx - dx_major)
    right_x = int(cx + dx_major)
    top_y = int(cy + dy_minor)  # Because OpenCV y-axis is downward

    # Define bounding box
    height, width = image.shape[:2]
    x1 = max(left_x - padding, 0)
    x2 = min(right_x + padding, width)
    y1 = max(top_y - padding, 0)
    y2 = height  # keep full height downward

    cropped = image[y1:y2, x1:x2]
    return cropped


def detect_elliptical_portafilter_with_holes(image):
    if image is None:
        print(f"Error loading image: {image}")
        return None, None
    output = image.copy()
    output2 = image.copy()

    # Check if image is too sharp or too blurry
    lap_var = cv2.Laplacian(image, cv2.CV_64F).var()
    print(f"Laplacian variance: {lap_var:.2f}")
    if lap_var > 500:
        print(f"⚠️ Image is too sharp (Laplacian variance: {lap_var:.2f})")
        image = cv2.medianBlur(image, 5)
    elif lap_var < 100:
        print(f"⚠️ Image is too blurry (Laplacian variance: {lap_var:.2f})")
        # Sharpening kernel
        sharpen_kernel = np.array([[0, -2, 0],
                                [-2, 9, -2],
                                [0, -2, 0]], dtype=np.float32)

        # Apply sharpening
        image = cv2.filter2D(image, -1, sharpen_kernel)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blurred image for Portafilter Ellipse Detection
    # el_gray = cv2.medianBlur(gray, 5)
    el_gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    el_edges = cv2.Canny(el_gray, 100, 200, apertureSize=3)
    cv2.imshow("og edges", edges)
    cv2.imshow("el edges", el_edges)

    # === 1. Hough Line Detection ===
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

    # cv2.imshow("Hough Lines", output)

    # === 2. Hough Circle Detection ===
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25,
    #                             param1=100, param2=30, minRadius=5, maxRadius=100)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for x, y, r in circles[0, :]:
    #         cv2.circle(output, (x, y), r, (255, 0, 0), 1)
    # cv2.imshow("Hough Cricles", output)
    circles = []

    # === 3. Ellipse Detection via Contours ===
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

    cv2.imshow("Hough Ellipse", output)


    # Step 4: Score and highlight best ellipse
    best_ellipse = ellipse_feature_score(image, lines,  circles[0] if len(circles) else [], ellipses, areas) # lines, circles[0] if len(circles) else []
    if best_ellipse is None and len(ellipses) > 0:
        print("⚠️ No high-score ellipse found — using largest fallback ellipse.")
        best_ellipse = max(
            ellipses,
            key=lambda el: el[1][0] * el[1][1]  # selects by width × height
        )
    # cropped_img = crop_image_by_ellipse(image, best_ellipse)
    #cv2.imwrite("cropped_portafilter.jpg", cropped_img)
    # cv2.imshow("Cropped Portafilter", cropped_img)

    safe_draw_ellipse(output2, best_ellipse, (0, 255, 255), 15)  # best in yellow

    return output2


img_path = os.path.join(FRAME_DIR, "frame_0000.jpg")
frame = load_image_with_orientation(img_path)
result = detect_elliptical_portafilter_with_holes(frame)
cv2.imshow("Detected Portafilter", result)
cv2.waitKey(0)