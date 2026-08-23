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

# ----------------- ALL FILTERS (ORIGINAL + EXTENDED) -----------------
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

# ----------------- PERSISTENT LIVE PORTAL WINDOW (MULTIPLE SIMULTANEOUS FILTERS) -----------------
class LivePortalWindow:
    """A locked 4-point spatial portal that renders its own live filter simultaneously with other portals."""
    def __init__(self, poly_pts, filter_name, portal_idx=1):
        self.poly_pts = np.copy(poly_pts)
        self.filter_name = filter_name
        self.portal_idx = portal_idx
        self.theme_color = FILTER_COLORS.get(filter_name, (0, 255, 255))
        self.x, self.y, self.bw, self.bh = cv2.boundingRect(self.poly_pts)
        self.created_at = time.time()

    def render(self, img, mask_person, frame_galaxy):
        h, w = img.shape[:2]
        x, y, bw, bh = self.x, self.y, self.bw, self.bh
        x, y = max(0, x), max(0, y)
        bw, bh = min(w - x, bw), min(h - y, bh)

        if bw > 0 and bh > 0:
            roi = img[y:y+bh, x:x+bw].copy()
            filtered_roi = apply_filter(roi, self.filter_name, x, y, mask_person, frame_galaxy)

            mask = np.zeros((bh, bw), dtype=np.uint8)
            poly_roi = self.poly_pts - [x, y]
            cv2.fillPoly(mask, [poly_roi], 255)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)

            # Glowing multi-layer neon border
            cv2.polylines(img, [self.poly_pts], True, self.theme_color, 2)
            cv2.polylines(img, [self.poly_pts], True, (255, 255, 255), 1)

            # Sparkles on perimeter
            for i in range(4):
                pt1 = self.poly_pts[i]
                pt2 = self.poly_pts[(i + 1) % 4]
                if random.random() < 0.7:
                    alpha = random.random()
                    spx = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + random.randint(-4, 4)
                    spy = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + random.randint(-4, 4)
                    cv2.circle(img, (spx, spy), random.randint(1, 3), (255, 255, 255), -1)

            # Holographic Filter Badge
            badge_text = f"🪟 {self.filter_name} #{self.portal_idx}"
            ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            top_y = min([p[1] for p in self.poly_pts])
            top_x = min([p[0] for p in self.poly_pts])
            bx = top_x
            by = max(25, top_y - 8)
            cv2.rectangle(img, (bx - 3, by - ts[1] - 3), (bx + ts[0] + 5, by + 3), (16, 16, 22), -1)
            cv2.rectangle(img, (bx - 3, by - ts[1] - 3), (bx + ts[0] + 5, by + 3), self.theme_color, 1)
            cv2.putText(img, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


# ----------------- FILTER PIPELINE -----------------
def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    h_r, w_r = roi.shape[:2]
    if h_r <= 0 or w_r <= 0:
        return roi

    t = time.time()

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
        synth[:, :, 0] = np.clip(synth[:, :, 0] * 1.4, 0, 255)
        synth[:, :, 2] = np.clip(synth[:, :, 2] * 1.5, 0, 255)
        synth[:, :, 1] = np.clip(synth[:, :, 1] * 0.4, 0, 255)
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


# ----------------- MAIN STUDIO -----------------
locked_portals = [] # List of LivePortalWindow active concurrently
last_lock_time = 0.0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    t = time.time()
    
    frame_galaxy = galaxy_bg[:h, :w]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = time.time_ns() // 1_000_000
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    filter_name = filters[current_filter]
    theme_col = FILTER_COLORS.get(filter_name, (0, 255, 255))
    
    mask_person = None
    if segmenter is not None:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    # ----------------- 1. RENDER ALL CONCURRENT LOCKED PORTALS SIMULTANEOUSLY -----------------
    for lp in locked_portals:
        lp.render(img, mask_person, frame_galaxy)

    pts_portal = []
    change_filter = False
    pinch_detected = False

    if results.hand_landmarks:
        # Cek gestur untuk ganti filter (Sentuh Telunjuk Kedua Tangan < 40px)
        if len(results.hand_landmarks) >= 2:
            idx0 = results.hand_landmarks[0][8]
            idx1 = results.hand_landmarks[1][8]
            pt0 = (int(idx0.x * w), int(idx0.y * h))
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 40:
                change_filter = True
                
        # Cek gestur ganti filter cadangan (Jempol & Kelingking) & PINCH untuk LOCK PORTAL
        for hand_lms in results.hand_landmarks:
            thumb = hand_lms[4]
            index = hand_lms[8]
            pinky = hand_lms[20]
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
            px, py = int(pinky.x * w), int(pinky.y * h)

            if math.hypot(tx - px, ty - py) < 40:
                change_filter = True

            # Pinch (Jempol & Telunjuk rapat < 35px)
            if math.hypot(tx - ix, ty - iy) < 35:
                pinch_detected = True
                
        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(filters)
                gesture_triggered = True
        else:
            gesture_triggered = False

        # ----------------- ALWAYS 4 POINTS USING TWO HANDS (THUMB & INDEX) -----------------
        # Exact 4-Point Geometry from Retrolens Original
        for hand_lms in results.hand_landmarks:
            for id_lm in [4, 8]:
                cx = int(hand_lms[id_lm].x * w)
                cy = int(hand_lms[id_lm].y * h)
                pts_portal.append([cx, cy])
                cv2.circle(img, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

        # 4 POIN -> PORTAL SEGIEMPAT (DUA TANGAN)
        if len(pts_portal) == 4:
            pts_portal.sort(key=lambda p: p[1])
            top_pts = pts_portal[:2]
            bottom_pts = pts_portal[2:]
            top_pts.sort(key=lambda p: p[0])
            bottom_pts.sort(key=lambda p: p[0])
            
            poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)
            
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
                
                cv2.polylines(img, [poly_pts], True, (255, 255, 255), 2)
                
                # --- PARTIKEL GLOW DI TEPI PORTAL ---
                for i in range(4):
                    pt1 = poly_pts[i]
                    pt2 = poly_pts[(i+1)%4]
                    for _ in range(5):
                        alpha = np.random.random()
                        px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
                        py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
                        cv2.circle(img, (px, py), np.random.randint(1, 4), (0, 255, 255), -1)

                cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], max(30, top_pts[0][1] - 10)), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

                # ----------------- PINCH TO LOCK NEW CONCURRENT LIVE PORTAL -----------------
                now = time.time()
                if pinch_detected and (now - last_lock_time > 0.8):
                    portal_idx = len(locked_portals) + 1
                    new_live_portal = LivePortalWindow(poly_pts, filter_name, portal_idx)
                    locked_portals.append(new_live_portal)
                    last_lock_time = now
                    
                    # Automatically advance filter for the NEXT portal you create!
                    current_filter = (current_filter + 1) % len(filters)
                    print(f"✨ LOCKED LIVE PORTAL #{portal_idx}: [{filter_name}]! Next Filter: [{filters[current_filter]}]")

    # ----------------- OVERLAY HUD & GUIDE -----------------
    cv2.putText(img, f"Filter Aktif: {filters[current_filter]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, theme_col, 2)
    cv2.putText(img, f"Portal Aktif Simultan: {len(locked_portals)}  |  👌 Pinch (Jempol+Telunjuk) = Kunci Portal Baru  |  'c' = Reset", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow('RETROLENS 4-Point Multi-Portal Studio', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        locked_portals.clear()
        print("🗑️ Semua portal dihapus.")
    elif key == ord('u'):
        if locked_portals:
            locked_portals.pop()
    elif key == ord('n'):
        current_filter = (current_filter + 1) % len(filters)

cap.release()
cv2.destroyAllWindows()