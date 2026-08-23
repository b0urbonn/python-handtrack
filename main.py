import os
import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_LANDMARKER_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')
FACE_LANDMARKER_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')
SELFIE_SEGMENTER_PATH = os.path.join(SCRIPT_DIR, 'selfie_segmenter.tflite')
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ----------------- FILTERS (ORIGINAL + EXTENDED) -----------------
FILTERS = [
    "GALAXY",
    "TOUCHDESIGNER",
    "NEON",
    "THERMAL",
    "DUAL-TONE",
    "GLITCH",
    "KALEIDOSCOPE",
    "PIXELATE",
    "MATRIX",
    "SKETCH",
    "INVERT",
    "SEPIA",
    "BLUR",
    "MONO"
]

# ----------------- CAPTURED FROZEN SNAPSHOT PICTURE (EMBEDDED IN FRONT) -----------------
class CapturedPortalPicture:
    """Represents an actual frozen picture snapshot captured at that exact moment with its filter,
    layered on top (in front) of older captures with a glowing polaroid/neon frame."""
    
    def __init__(self, poly_pts, filter_name, snapshot_roi, mask_crop):
        self.poly_pts = np.copy(poly_pts)
        self.filter_name = filter_name
        self.x, self.y, self.bw, self.bh = cv2.boundingRect(self.poly_pts)
        self.snapshot_roi = snapshot_roi.copy() # Actual frozen photo snapshot
        self.mask_crop = mask_crop.copy()
        self.mask_3ch = cv2.cvtColor(self.mask_crop, cv2.COLOR_GRAY2BGR)
        self.created_at = time.time()
        self.border_phase = random.uniform(0, math.pi * 2)
        self.id_num = random.randint(100, 999)

    def draw(self, img):
        h, w = img.shape[:2]
        x, y, bw, bh = self.x, self.y, self.bw, self.bh
        x, y = max(0, x), max(0, y)
        bw, bh = min(w - x, bw), min(h - y, bh)

        if bw > 0 and bh > 0 and self.snapshot_roi.shape[0] >= bh and self.snapshot_roi.shape[1] >= bw:
            roi = img[y:y+bh, x:x+bw]
            snap = self.snapshot_roi[:bh, :bw]
            m3 = self.mask_3ch[:bh, :bw]

            # 1. Paste the actual frozen picture snapshot in front
            img[y:y+bh, x:x+bw] = np.where(m3 == 255, snap, roi)

            # 2. Glowing Neon / Polaroid Border
            cv2.polylines(img, [self.poly_pts], True, (0, 255, 255), 2)
            cv2.polylines(img, [self.poly_pts], True, (255, 255, 255), 1)

            # Corner Framing Brackets
            bk = min(14, min(bw // 3, bh // 3))
            for (bx, by, dx, dy) in [(x, y, 1, 1), (x + bw, y, -1, 1), (x, y + bh, 1, -1), (x + bw, y + bh, -1, -1)]:
                cv2.line(img, (bx, by), (bx + dx * bk, by), (255, 255, 255), 2)
                cv2.line(img, (bx, by), (bx, by + dy * bk), (255, 255, 255), 2)

            # 3. Hologram Photo Filter Badge
            badge_text = f"📷 {self.filter_name}"
            ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bx1 = x + 4
            by1 = max(25, y - 6)
            cv2.rectangle(img, (bx1 - 3, by1 - ts[1] - 3), (bx1 + ts[0] + 5, by1 + 3), (18, 18, 24), -1)
            cv2.rectangle(img, (bx1 - 3, by1 - ts[1] - 3), (bx1 + ts[0] + 5, by1 + 3), (0, 255, 255), 1)
            cv2.putText(img, badge_text, (bx1, by1), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


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
        filtered[mask_c == 255] = [0, 165, 255]  # Orange
        filtered[mask_c == 0] = [147, 20, 255]   # Pink
        return filtered

    elif filter_name == "PIXELATE":
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r // 12), max(1, h_r // 12)), interpolation=cv2.INTER_LINEAR)
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
        stride = 16
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

    elif filter_name == "MATRIX":
        matrix_roi = np.zeros_like(roi)
        for gy in range(0, h_r, 14):
            cv2.line(matrix_roi, (0, gy), (w_r, gy), (0, 255, 128), 1)
        for gx in range(0, w_r, 18):
            cv2.line(matrix_roi, (gx, 0), (gx, h_r), (0, 255, 128), 1)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        matrix_roi[edges > 0] = (255, 255, 255)
        return cv2.addWeighted(roi, 0.3, matrix_roi, 0.7, 0)

    return roi


# ----------------- MAIN STUDIO -----------------
def main():
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Initialize Hand Landmarker
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    landmarker = HandLandmarker.create_from_options(hand_options)

    # Initialize Selfie Segmenter
    segmenter = None
    try:
        ImageSegmenter = mp.tasks.vision.ImageSegmenter
        ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
        seg_options = ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=SELFIE_SEGMENTER_PATH),
            running_mode=VisionRunningMode.VIDEO,
            output_category_mask=True
        )
        segmenter = ImageSegmenter.create_from_options(seg_options)
    except Exception as e:
        print("Segmenter status:", e)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera.")
        return

    # Galaxy Background Initialization
    galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    galaxy_bg[:] = (30, 10, 40)
    for _ in range(800):
        sx = np.random.randint(0, 1920)
        sy = np.random.randint(0, 1080)
        galaxy_bg[sy, sx] = (255, 255, 255)
    for _ in range(100):
        sx = np.random.randint(0, 1920)
        sy = np.random.randint(0, 1080)
        cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6), (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)

    print("✨ RETROLENS ORIGINAL PORTAL & FROZEN PICTURE CAPTURE ACTIVE ✨")

    current_filter = 0
    gesture_triggered = False
    last_capture_time = 0.0
    last_timestamp_ms = 0
    flash_timer = 0
    captured_pictures = [] # List of CapturedPortalPicture (ordered back-to-front)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        frame_galaxy = galaxy_bg[:h, :w]

        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        filter_name = FILTERS[current_filter]

        mask_person = None
        if segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

        # ----------------- 1. DRAW ALL FROZEN CAPTURED PICTURES (IN ORDER, NEWEST IN FRONT) -----------------
        for cp in captured_pictures:
            cp.draw(img)

        pts_portal = []
        change_filter = False
        pinch_detected = False

        if results.hand_landmarks:
            # 1. Cek Gestur Ganti Filter: Sentuh Ujung Telunjuk Kedua Tangan (< 40px)
            if len(results.hand_landmarks) >= 2:
                idx0 = results.hand_landmarks[0][8]
                idx1 = results.hand_landmarks[1][8]
                pt0 = (int(idx0.x * w), int(idx0.y * h))
                pt1 = (int(idx1.x * w), int(idx1.y * h))
                if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 40:
                    change_filter = True

            # 2. Cek Gestur Jempol & Kelingking (Ganti Filter) & Pinch Jempol-Telunjuk (Capture Photo)
            for hand_lms in results.hand_landmarks:
                thumb = hand_lms[4]
                index = hand_lms[8]
                pinky = hand_lms[20]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                px, py = int(pinky.x * w), int(pinky.y * h)

                # Gestur ganti filter: Jempol & Kelingking
                if math.hypot(tx - px, ty - py) < 40:
                    change_filter = True

                # Gestur PINCH untuk CAPTURE: Jempol & Telunjuk rapat (< 35px)
                if math.hypot(tx - ix, ty - iy) < 35:
                    pinch_detected = True

            if change_filter:
                if not gesture_triggered:
                    current_filter = (current_filter + 1) % len(FILTERS)
                    gesture_triggered = True
            else:
                gesture_triggered = False

            # Ambil 4 titik (Jempol & Telunjuk dari kedua tangan) persis seperti original
            for hand_lms in results.hand_landmarks:
                for id_lm in [4, 8]:
                    cx_lm = int(hand_lms[id_lm].x * w)
                    cy_lm = int(hand_lms[id_lm].y * h)
                    pts_portal.append([cx_lm, cy_lm])
                    cv2.circle(img, (cx_lm, cy_lm), 8, (255, 255, 0), cv2.FILLED)

            # ----------------- 4 POIN -> LIVE PORTAL SEGIEMPAT (PERSIS ORIGINAL) -----------------
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
                            spx = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
                            spy = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
                            cv2.circle(img, (spx, spy), np.random.randint(1, 4), (0, 255, 255), -1)

                    cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], max(30, top_pts[0][1] - 10)), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

                    # ----------------- CAPTURE FROZEN PICTURE ON PINCH (EMBEDDED IN FRONT) -----------------
                    # Takes an actual photo snapshot of that exact moment with its filter, embedded in front!
                    now = time.time()
                    if pinch_detected and (now - last_capture_time > 0.8):
                        new_picture = CapturedPortalPicture(poly_pts, filter_name, filtered_roi, mask)
                        captured_pictures.append(new_picture) # Added in front!
                        last_capture_time = now
                        flash_timer = 2

                        # Automatic filter rotation for next capture
                        current_filter = (current_filter + 1) % len(FILTERS)
                        print(f"📸 CAPTURED FROZEN PICTURE #{len(captured_pictures)} [{filter_name}] EMBEDDED IN FRONT! Next: [{FILTERS[current_filter]}]")

        # Shutter Flash on Capture
        if flash_timer > 0:
            flash_overlay = np.full_like(img, 255)
            cv2.addWeighted(flash_overlay, 0.4, img, 0.6, 0, img)
            flash_timer -= 1

        # ----------------- HUD & INSTRUCTIONS -----------------
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        hud_line1 = f"FILTER AKTIF: {filter_name}  |  FOTO TERSIMPAN: {len(captured_pictures)}"
        hud_line2 = "📸 PINCH (Jempol+Telunjuk) = Capture Frozen Picture In Front  |  🤙 Jempol+Kelingking = Ganti Filter  |  'c' = Hapus Semua"
        cv2.putText(img, hud_line1, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(img, hud_line2, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.imshow('RETROLENS Pake Python - Photo Snapshot Portal Capture', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_pictures.clear()
            print("🗑️ Semua foto dihapus.")
        elif key == ord('n'):
            current_filter = (current_filter + 1) % len(FILTERS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()