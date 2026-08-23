import os
import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(SCRIPT_DIR, 'hand_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.65,
    min_hand_presence_confidence=0.65,
    min_tracking_confidence=0.65
)
landmarker = HandLandmarker.create_from_options(options)

segmenter = None
try:
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    seg_options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(SCRIPT_DIR, 'selfie_segmenter.tflite')),
        running_mode=VisionRunningMode.VIDEO,
        output_category_mask=True
    )
    segmenter = ImageSegmenter.create_from_options(seg_options)
except Exception as e:
    print("Segmenter tidak tersedia (pastikan file selfie_segmenter.tflite ada)", e)

cap = cv2.VideoCapture(0)

# Generate Galaxy Background
galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8) 
galaxy_bg[:] = (30, 10, 40) # Space purple
for _ in range(800):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    galaxy_bg[sy, sx] = (255, 255, 255)
for _ in range(100):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6), (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)

print("Membuka Kamera... Tekan tombol 'q' di keyboard untuk keluar.")

# ----------------- ALL ORIGINAL FILTERS + BRAND NEW EXPANDED FILTERS -----------------
filters = [
    "GALAXY",
    "TOUCHDESIGNER",
    "CYBER-MATRIX",
    "NEON",
    "THERMAL",
    "KALEIDOSCOPE",
    "80s-SYNTHWAVE",
    "CARTOON-TOON",
    "VORTEX-WARP",
    "EDGE-GLOW",
    "DUAL-TONE",
    "PIXELATE",
    "NIGHT-VISION",
    "INVERT",
    "SEPIA",
    "BLUR",
    "SKETCH",
    "GLITCH",
    "MONO"
]
current_filter = 0
gesture_triggered = False

FILTER_COLORS = {
    "GALAXY": (255, 200, 0),
    "TOUCHDESIGNER": (203, 19, 255),
    "CYBER-MATRIX": (0, 255, 128),
    "NEON": (255, 255, 0),
    "THERMAL": (0, 140, 255),
    "KALEIDOSCOPE": (255, 0, 200),
    "80s-SYNTHWAVE": (255, 60, 180),
    "CARTOON-TOON": (0, 220, 255),
    "VORTEX-WARP": (180, 0, 255),
    "EDGE-GLOW": (0, 255, 255),
    "DUAL-TONE": (0, 165, 255),
    "PIXELATE": (50, 255, 50),
    "NIGHT-VISION": (20, 255, 60),
    "INVERT": (0, 255, 255),
    "SEPIA": (0, 180, 220),
    "BLUR": (180, 200, 255),
    "SKETCH": (220, 220, 220),
    "GLITCH": (255, 0, 128),
    "MONO": (200, 200, 200)
}

def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    h_r, w_r = roi.shape[:2]
    if h_r <= 0 or w_r <= 0:
        return roi

    t = time.time()

    # Original Filters
    if filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255] # Orange
        filtered[mask_c == 0] = [147, 20, 255]  # Pink
        return filtered
    elif filter_name == "PIXELATE":
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r // 10), max(1, h_r // 10)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "GLITCH":
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y+bh, x:x+bw]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]
        bg_condition = (roi_mask == 0)
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered
    elif filter_name == "TOUCHDESIGNER":
        td_roi = np.zeros_like(roi)
        stride = 14
        nodes = []
        for gy in range(0, h_r, stride):
            for gx in range(0, w_r, stride):
                px = gx + int(math.sin(gy * 0.05 + t * 3.0) * 3)
                py = gy + int(math.cos(gx * 0.05 + t * 3.0) * 3)
                nodes.append((px, py))
        for i in range(0, len(nodes), 2):
            for j in range(i + 1, min(i + 4, len(nodes))):
                p1, p2 = nodes[i], nodes[j]
                if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < 28:
                    cv2.line(td_roi, p1, p2, (203, 19, 255), 1)
        for p in nodes:
            cv2.circle(td_roi, p, 2, (0, 255, 255), -1)
        return cv2.addWeighted(roi, 0.4, td_roi, 0.6, 0)

    # ----------------- NEW ADDED FILTERS -----------------
    elif filter_name == "CYBER-MATRIX":
        matrix_roi = np.zeros_like(roi)
        for gy in range(0, h_r, 14):
            cv2.line(matrix_roi, (0, gy), (w_r, gy), (0, 255, 128), 1)
        for gx in range(0, w_r, 18):
            cv2.line(matrix_roi, (gx, 0), (gx, h_r), (0, 255, 128), 1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        matrix_roi[edges > 0] = (255, 255, 255)
        return cv2.addWeighted(roi, 0.3, matrix_roi, 0.7, 0)

    elif filter_name == "KALEIDOSCOPE":
        quad = roi[:h_r//2, :w_r//2]
        if quad.shape[0] > 0 and quad.shape[1] > 0:
            quad_flip_h = cv2.flip(quad, 1)
            top_half = np.hstack((quad, quad_flip_h))
            bottom_half = cv2.flip(top_half, 0)
            kaleido = np.vstack((top_half, bottom_half))
            if kaleido.shape[:2] != (h_r, w_r):
                kaleido = cv2.resize(kaleido, (w_r, h_r))
            hsv = cv2.cvtColor(kaleido, cv2.COLOR_BGR2HSV)
            shift_hue = int((t * 40) % 180)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int32) + shift_hue) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    elif filter_name == "80s-SYNTHWAVE":
        synth = roi.copy()
        synth[:, :, 0] = np.clip(synth[:, :, 0] * 1.4, 0, 255) # Blue
        synth[:, :, 2] = np.clip(synth[:, :, 2] * 1.5, 0, 255) # Red
        synth[:, :, 1] = np.clip(synth[:, :, 1] * 0.4, 0, 255) # Green
        for sy in range(0, h_r, 6):
            synth[sy, :] = synth[sy, :] // 2
        return synth

    elif filter_name == "CARTOON-TOON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(roi, 9, 250, 250)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(color, edges_3ch)

    elif filter_name == "VORTEX-WARP":
        cx, cy = w_r / 2.0, h_r / 2.0
        y_grid, x_grid = np.indices((h_r, w_r), dtype=np.float32)
        dx = x_grid - cx
        dy = y_grid - cy
        radius = np.sqrt(dx*dx + dy*dy)
        max_r = max(1.0, min(cx, cy))
        angle = np.arctan2(dy, dx)
        vortex_angle = angle + (1.0 - np.clip(radius / max_r, 0, 1)) * 1.8 * math.sin(t * 3.0)
        map_x = np.clip(cx + radius * np.cos(vortex_angle), 0, w_r - 1).astype(np.float32)
        map_y = np.clip(cy + radius * np.sin(vortex_angle), 0, h_r - 1).astype(np.float32)
        return cv2.remap(roi, map_x, map_y, cv2.INTER_LINEAR)

    elif filter_name == "EDGE-GLOW":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.uint8(np.clip(np.sqrt(sobelx**2 + sobely**2), 0, 255))
        glow = cv2.applyColorMap(mag, cv2.COLORMAP_HOT)
        return cv2.addWeighted(roi, 0.4, glow, 0.6, 0)

    elif filter_name == "NIGHT-VISION":
        green_roi = roi.copy()
        green_roi[:, :, 0] = 0
        green_roi[:, :, 2] = 0
        green_roi[:, :, 1] = np.clip(green_roi[:, :, 1] * 1.8, 0, 255)
        noise = np.random.randint(0, 30, (h_r, w_r), dtype=np.uint8)
        green_roi[:, :, 1] = cv2.add(green_roi[:, :, 1], noise)
        return green_roi
        
    return roi

def lm_dist(lm1, lm2):
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def is_finger_raised(hand_lms, finger_id):
    """Accurately checks if a specific finger is raised/extended vs curled/closed."""
    w_lm = hand_lms[0] # Wrist

    if finger_id == 4: # Thumb
        d_tip_pinky = lm_dist(hand_lms[4], hand_lms[17])
        d_ip_pinky = lm_dist(hand_lms[3], hand_lms[17])
        d_tip_wrist = lm_dist(hand_lms[4], w_lm)
        d_mcp_wrist = lm_dist(hand_lms[2], w_lm)
        return (d_tip_pinky > d_ip_pinky * 1.12) and (d_tip_wrist > d_mcp_wrist * 1.10)

    elif finger_id == 8: # Index
        return (lm_dist(hand_lms[8], w_lm) > lm_dist(hand_lms[6], w_lm) * 1.12) and (lm_dist(hand_lms[8], w_lm) > lm_dist(hand_lms[5], w_lm) * 1.20)

    elif finger_id == 12: # Middle
        return (lm_dist(hand_lms[12], w_lm) > lm_dist(hand_lms[10], w_lm) * 1.12) and (lm_dist(hand_lms[12], w_lm) > lm_dist(hand_lms[9], w_lm) * 1.20)

    elif finger_id == 16: # Ring
        return (lm_dist(hand_lms[16], w_lm) > lm_dist(hand_lms[14], w_lm) * 1.12) and (lm_dist(hand_lms[16], w_lm) > lm_dist(hand_lms[13], w_lm) * 1.20)

    elif finger_id == 20: # Pinky
        return (lm_dist(hand_lms[20], w_lm) > lm_dist(hand_lms[18], w_lm) * 1.12) and (lm_dist(hand_lms[20], w_lm) > lm_dist(hand_lms[17], w_lm) * 1.20)

    return False

def draw_finger_layers(img, pt, theme_color, t):
    """Draws multi-layered concentric ripple halos and glowing core at each connected RAISED finger point."""
    x, y = pt
    # Layer 1: Outer expanding ripple halo
    r1 = int(15 + 4 * math.sin(t * 5.0 + x * 0.01))
    cv2.circle(img, (x, y), r1, theme_color, 1)
    
    # Layer 2: Mid glowing halo
    r2 = int(10 + 2 * math.sin(t * 7.0 + y * 0.01))
    cv2.circle(img, (x, y), r2, (0, 255, 255), 2)
    
    # Layer 3: Solid white-hot core
    cv2.circle(img, (x, y), 5, (255, 255, 255), -1)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    t = time.time()
    
    # Crop galaxy bg to match frame size dynamically
    frame_galaxy = galaxy_bg[:h, :w]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = time.time_ns() // 1_000_000
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    filter_name = filters[current_filter]
    theme_col = FILTER_COLORS.get(filter_name, (0, 255, 255))
    
    mask_person = None
    if filter_name == "GALAXY" and segmenter is not None:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    change_filter = False
    pts_portal = []
    both_hands_ready = False

    if results.hand_landmarks:
        # Cek gestur untuk ganti filter (Sentuh Telunjuk Kedua Tangan < 40px)
        if len(results.hand_landmarks) >= 2:
            idx0 = results.hand_landmarks[0][8]
            idx1 = results.hand_landmarks[1][8]
            pt0 = (int(idx0.x * w), int(idx0.y * h))
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 40:
                change_filter = True
                
        # Cek gestur ganti filter cadangan (Jempol & Kelingking)
        for hand_lms in results.hand_landmarks:
            thumb = hand_lms[4]
            pinky = hand_lms[20]
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            px, py = int(pinky.x * w), int(pinky.y * h)
            if math.hypot(tx - px, ty - py) < 40:
                change_filter = True
                
        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(filters)
                gesture_triggered = True
        else:
            gesture_triggered = False

        # ----------------- ALWAYS 2 HANDS REQUIRED TO CREATE A PORTAL -----------------
        if len(results.hand_landmarks) >= 2:
            h1_lms = results.hand_landmarks[0]
            h2_lms = results.hand_landmarks[1]

            fingertip_ids = [4, 8, 12, 16, 20]
            h1_raised = []
            h2_raised = []

            for fid in fingertip_ids:
                if is_finger_raised(h1_lms, fid):
                    h1_raised.append([int(h1_lms[fid].x * w), int(h1_lms[fid].y * h)])
                if is_finger_raised(h2_lms, fid):
                    h2_raised.append([int(h2_lms[fid].x * w), int(h2_lms[fid].y * h)])

            # Only open the portal when BOTH hands have at least one raised finger
            if len(h1_raised) >= 1 and len(h2_raised) >= 1:
                both_hands_ready = True
                pts_portal = h1_raised + h2_raised

                # Draw multi-layer halos on every connected RAISED fingertip point
                for pt in pts_portal:
                    draw_finger_layers(img, tuple(pt), theme_col, t)

                # ----------------- MULTI-LAYER PORTAL GEOMETRY -----------------
                poly_pts = None

                if len(pts_portal) >= 3:
                    pts_array = np.array(pts_portal, dtype=np.int32)
                    hull = cv2.convexHull(pts_array)
                    poly_pts = hull.reshape((-1, 2))

                elif len(pts_portal) == 2:
                    # 2 Raised Fingers (1 per hand) -> Expand into a dynamic 4-point pill/capsule portal
                    p1 = np.array(pts_portal[0], dtype=np.float32)
                    p2 = np.array(pts_portal[1], dtype=np.float32)
                    v = p2 - p1
                    v_len = np.linalg.norm(v)
                    if v_len > 15:
                        normal = np.array([-v[1], v[0]]) / v_len
                        thickness = 50.0
                        c1 = p1 + normal * thickness
                        c2 = p2 + normal * thickness
                        c3 = p2 - normal * thickness
                        c4 = p1 - normal * thickness
                        poly_pts = np.int32([c1, c2, c3, c4])

                if poly_pts is not None and len(poly_pts) >= 3:
                    x, y, bw, bh = cv2.boundingRect(poly_pts)
                    x, y = max(0, x), max(0, y)
                    bw, bh = min(w - x, bw), min(h - y, bh)
                    
                    if bw > 0 and bh > 0:
                        roi = img[y:y+bh, x:x+bw].copy()
                        filtered_roi = apply_filter(roi, filter_name, x, y, mask_person, frame_galaxy)
                        
                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        poly_roi = poly_pts - [x, y]
                        cv2.fillPoly(mask, [poly_roi], 255)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        
                        img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)
                        
                        # --- MULTI-LAYER PORTAL PERIMETER LAYERS ---
                        # Layer 1: Outer Glowing Neon Contour
                        cv2.polylines(img, [poly_pts], True, theme_col, 3)
                        
                        # Layer 2: Sharp Inner White Event Horizon
                        cv2.polylines(img, [poly_pts], True, (255, 255, 255), 1)
                        
                        # Layer 3: Nested Concentric Inset Horizon Layer
                        poly_center = np.mean(poly_pts, axis=0)
                        inset_pts = np.int32(poly_center + (poly_pts - poly_center) * 0.93)
                        cv2.polylines(img, [inset_pts], True, (0, 255, 255), 1)

                        # Cross-energy lattice lines connecting finger nodes across the portal
                        for i in range(0, len(poly_pts), 2):
                            for j in range(i + 2, len(poly_pts), 2):
                                p1 = tuple(poly_pts[i])
                                p2 = tuple(poly_pts[j])
                                cv2.line(img, p1, p2, (int(theme_col[0] * 0.5), int(theme_col[1] * 0.5), int(theme_col[2] * 0.5)), 1)
                        
                        # --- PARTIKEL GLOW & SPARKS DI TEPI PORTAL ---
                        num_pts = len(poly_pts)
                        for i in range(num_pts):
                            pt1 = poly_pts[i]
                            pt2 = poly_pts[(i + 1) % num_pts]
                            for _ in range(3):
                                alpha = np.random.random()
                                px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-8, 8)
                                py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-8, 8)
                                cv2.circle(img, (px, py), np.random.randint(1, 3), (255, 255, 255), -1)

                        # Portal Title Label
                        top_y = min([p[1] for p in poly_pts])
                        top_x = min([p[0] for p in poly_pts])
                        cv2.putText(img, f"PORTAL: {filter_name} ({len(pts_portal)} JARI AKTIF)", (top_x, max(30, top_y - 12)), 
                                    cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 2)
        else:
            # Only 1 hand detected -> draw glow dots and guide user to raise 2nd hand
            for hand_lms in results.hand_landmarks:
                for fid in [4, 8, 12, 16, 20]:
                    if is_finger_raised(hand_lms, fid):
                        fx = int(hand_lms[fid].x * w)
                        fy = int(hand_lms[fid].y * h)
                        draw_finger_layers(img, (fx, fy), theme_col, t)

    # ----------------- OVERLAY HUD & GUIDE -----------------
    cv2.putText(img, f"Filter Aktif: {filters[current_filter]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_col, 2)
    if both_hands_ready:
        status_msg = f"🌀 PORTAL AKTIF: {len(pts_portal)} Jari Terangkat dari 2 Tangan"
        cv2.putText(img, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        status_msg = "🖐️ Angkat 2 Tangan & Jari untuk Membuka Portal"
        cv2.putText(img, status_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow('RETROLENS 2-Hand Raised-Finger Multi-Layer Portal', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)

cap.release()
cv2.destroyAllWindows()