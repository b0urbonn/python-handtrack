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

# ----------------- FILTERS & THEMES -----------------
FILTERS = [
    {"name": "COSMIC GALAXY 🌌", "id": "GALAXY", "theme_color": (255, 200, 0)},
    {"name": "TOUCHDESIGNER POP 🔮", "id": "TOUCHDESIGNER", "theme_color": (203, 19, 255)},
    {"name": "NEON LASER 💎", "id": "NEON", "theme_color": (255, 255, 0)},
    {"name": "PREDATOR THERMAL 🧊", "id": "THERMAL", "theme_color": (0, 140, 255)},
    {"name": "DUAL-TONE COMIC 🎭", "id": "DUALTONE", "theme_color": (0, 165, 255)},
    {"name": "CYBER MATRIX ⚡", "id": "MATRIX", "theme_color": (0, 255, 128)},
    {"name": "RGB GLITCH 📺", "id": "GLITCH", "theme_color": (255, 0, 128)},
    {"name": "KALEIDOSCOPE 🌈", "id": "KALEIDOSCOPE", "theme_color": (255, 0, 200)},
    {"name": "8-BIT PIXEL 🎮", "id": "PIXELATE", "theme_color": (50, 255, 50)},
    {"name": "PENCIL SKETCH ✏️", "id": "SKETCH", "theme_color": (220, 220, 220)},
    {"name": "INVERTED X-RAY 👁️", "id": "INVERT", "theme_color": (0, 255, 255)},
    {"name": "SEPIA CINEMA 🎞️", "id": "SEPIA", "theme_color": (0, 180, 220)},
    {"name": "ETHEREAL BLUR 🌫️", "id": "BLUR", "theme_color": (180, 200, 255)},
    {"name": "MONO NOIR 🎬", "id": "MONO", "theme_color": (200, 200, 200)}
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm
]


# ----------------- VISUAL SHOCKWAVE -----------------
class Shockwave:
    def __init__(self, x, y, color=(0, 255, 255), max_radius=85):
        self.x = int(x)
        self.y = int(y)
        self.color = color
        self.radius = 6.0
        self.max_radius = float(max_radius)
        self.life = 1.0

    def update(self):
        self.radius += 5.5
        self.life = max(0.0, 1.0 - (self.radius / self.max_radius))
        return self.life > 0

    def draw(self, img):
        if self.life <= 0:
            return
        alpha = self.life
        color = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
        cv2.circle(img, (self.x, self.y), int(self.radius), color, max(1, int(3 * alpha)))


# ----------------- PARTICLE ENGINE -----------------
class Particle:
    def __init__(self, x, y, vx, vy, color, size, life, decay=0.96, p_type="spark"):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.color = color
        self.size = float(size)
        self.life = float(life)
        self.max_life = float(life)
        self.decay = decay
        self.p_type = p_type

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= self.decay
        self.vy *= self.decay
        if self.p_type == "firework":
            self.vy += 0.18
        self.life -= 1.0
        self.size = max(0.5, self.size * 0.95)
        return self.life > 0

    def draw(self, img):
        if self.life <= 0:
            return
        alpha = min(1.0, max(0.0, self.life / self.max_life))
        c = self.color
        color = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha)) if isinstance(c, tuple) else (255, 255, 255)
        pt = (int(self.x), int(self.y))
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(img, pt, max(1, int(self.size)), color, -1)


class ParticleManager:
    def __init__(self):
        self.particles = []
        self.shockwaves = []

    def add_particle(self, p):
        if len(self.particles) < 700:
            self.particles.append(p)

    def add_shockwave(self, x, y, color=(0, 255, 255), max_radius=85):
        self.shockwaves.append(Shockwave(x, y, color, max_radius))

    def emit_sparkles(self, x, y, color=(0, 255, 255), count=3, speed=3):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(1, speed)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, color, random.uniform(2, 5), random.randint(15, 30)))

    def emit_fireworks(self, x, y, count=35, color=None):
        colors = [(0, 165, 255), (255, 255, 0), (255, 50, 50), (50, 255, 50), (255, 0, 255), (0, 255, 255)]
        for _ in range(count):
            c = color if color else random.choice(colors)
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(3, 11)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, c, random.uniform(3, 6), random.randint(20, 45), decay=0.98, p_type="firework"))
        self.add_shockwave(x, y, color if color else (255, 255, 255), max_radius=85)

    def update_and_draw(self, img):
        self.particles = [p for p in self.particles if p.update() and (p.draw(img) or True)]
        self.shockwaves = [sw for sw in self.shockwaves if sw.update() and (sw.draw(img) or True)]


# ----------------- CAPTURED FROZEN SNAPSHOT PICTURE (EMBEDDED IN FRONT) -----------------
class CapturedPortalPicture:
    """A photographic snapshot frozen at that exact moment, styled as a premium glass holographic card layered in front."""
    
    def __init__(self, x1, y1, x2, y2, filter_info, snapshot_crop):
        self.x1 = int(min(x1, x2))
        self.y1 = int(min(y1, y2))
        self.x2 = int(max(x1, x2))
        self.y2 = int(max(y1, y2))
        self.w = max(50, self.x2 - self.x1)
        self.h = max(50, self.y2 - self.y1)
        self.filter_info = filter_info
        self.filter_name = filter_info["name"]
        self.theme_color = filter_info["theme_color"]
        self.snapshot_crop = snapshot_crop.copy()
        self.created_at = time.time()
        self.border_phase = random.uniform(0, math.pi * 2)

    def draw(self, img):
        h, w = img.shape[:2]
        sx1, sx2 = max(0, self.x1), min(w, self.x2)
        sy1, sy2 = max(0, self.y1), min(h, self.y2)
        cw, ch = sx2 - sx1, sy2 - sy1

        if cw > 0 and ch > 0:
            crop_resized = cv2.resize(self.snapshot_crop, (self.w, self.h))
            # Paste the exact frozen photo in front
            img[sy1:sy2, sx1:sx2] = crop_resized[:ch, :cw]

            # Glowing Holographic Border
            col = self.theme_color
            t = time.time()
            pulse = 0.85 + 0.15 * math.sin(t * 3.5 + self.border_phase)
            border_col = (int(col[0] * pulse), int(col[1] * pulse), int(col[2] * pulse))

            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), border_col, 2)
            cv2.rectangle(img, (sx1 + 2, sy1 + 2), (sx2 - 2, sy2 - 2), (255, 255, 255), 1)

            # High-Tech Corner Brackets
            bk = min(16, min(cw // 3, ch // 3))
            for (bx, by, dx, dy) in [(sx1, sy1, 1, 1), (sx2, sy1, -1, 1), (sx1, sy2, 1, -1), (sx2, sy2, -1, -1)]:
                cv2.line(img, (bx, by), (bx + dx * bk, by), (255, 255, 255), 3)
                cv2.line(img, (bx, by), (bx, by + dy * bk), (255, 255, 255), 3)

            # Clean Hologram Title Badge
            badge_text = f"📷 {self.filter_name.split()[0]}"
            ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)[0]
            bx1 = sx1 + 6
            by1 = max(30, sy1 - 8)
            cv2.rectangle(img, (bx1 - 4, by1 - ts[1] - 4), (bx1 + ts[0] + 6, by1 + 4), (16, 16, 22), -1)
            cv2.rectangle(img, (bx1 - 4, by1 - ts[1] - 4), (bx1 + ts[0] + 6, by1 + 4), border_col, 1)
            cv2.putText(img, badge_text, (bx1, by1), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1)


# ----------------- FILTER PIPELINE -----------------
def apply_filter(roi, filter_id, x=0, y=0, mask_person=None, frame_galaxy=None):
    h_r, w_r = roi.shape[:2]
    if h_r <= 0 or w_r <= 0:
        return roi

    t = time.time()

    if filter_id == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif filter_id == "INVERT":
        return cv2.bitwise_not(roi)

    elif filter_id == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)

    elif filter_id == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    elif filter_id == "DUALTONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [203, 19, 255]
        return filtered

    elif filter_id == "PIXELATE":
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r // 12), max(1, h_r // 12)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)

    elif filter_id == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)

    elif filter_id == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    elif filter_id == "GLITCH":
        shift = max(6, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi

    elif filter_id == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)

    elif filter_id == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y+bh, x:x+bw]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]
        bg_condition = (roi_mask == 0)
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered

    elif filter_id == "TOUCHDESIGNER":
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

    elif filter_id == "KALEIDOSCOPE":
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

    elif filter_id == "MATRIX":
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
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65
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

    # Galaxy Background
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

    print("✨ ULTRA-AESTHETIC PORTAL STUDIO: WHOLE-FINGER CONNECTIVITY & PHOTO EMBED ✨")

    current_filter_idx = 0
    gesture_triggered = False
    last_capture_time = 0.0
    last_timestamp_ms = 0
    flash_timer = 0
    captured_pictures = [] # Ordered back-to-front (newest in front)
    
    # Smooth Bounding Box Interpolator
    smooth_box = None

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

        cur_filter = FILTERS[current_filter_idx]

        mask_person = None
        if segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

        # ----------------- 1. DRAW ALL CAPTURED PICTURES IN FRONT (NEWEST ON TOP) -----------------
        for cp in captured_pictures:
            cp.draw(img)

        # ----------------- 2. WHOLE-FINGER TRACKING & ENERGY TETHERS -----------------
        all_fingertips = []
        all_hand_points = []
        pinch_detected = False
        change_filter = False

        if results.hand_landmarks:
            for hand_lms in results.hand_landmarks:
                # 5 Fingertips: Thumb (4), Index (8), Middle (12), Ring (16), Pinky (20)
                fingertip_ids = [4, 8, 12, 16, 20]
                for fid in fingertip_ids:
                    fx = int(hand_lms[fid].x * w)
                    fy = int(hand_lms[fid].y * h)
                    all_fingertips.append((fx, fy))

                # All 21 landmarks
                for lm in hand_lms:
                    all_hand_points.append((int(lm.x * w), int(lm.y * h)))

                # Draw skeleton connection lines
                for c1, c2 in HAND_CONNECTIONS:
                    p1 = (int(hand_lms[c1].x * w), int(hand_lms[c1].y * h))
                    p2 = (int(hand_lms[c2].x * w), int(hand_lms[c2].y * h))
                    cv2.line(img, p1, p2, (0, 80, 80), 3)
                    cv2.line(img, p1, p2, (0, 255, 255), 1)

                # Check Pinch / Capture Gestures:
                tx, ty = int(hand_lms[4].x * w), int(hand_lms[4].y * h)
                ix, iy = int(hand_lms[8].x * w), int(hand_lms[8].y * h)
                px, py = int(hand_lms[20].x * w), int(hand_lms[20].y * h)

                # Pinch (Thumb to Index) < 38px
                if math.hypot(tx - ix, ty - iy) < 38:
                    pinch_detected = True

                # Switch Filter (Thumb to Pinky) < 38px
                if math.hypot(tx - px, ty - py) < 38:
                    change_filter = True

            # Two hands index touch -> Switch filter
            if len(results.hand_landmarks) >= 2:
                h0_ix, h0_iy = int(results.hand_landmarks[0][8].x * w), int(results.hand_landmarks[0][8].y * h)
                h1_ix, h1_iy = int(results.hand_landmarks[1][8].x * w), int(results.hand_landmarks[1][8].y * h)
                if math.hypot(h0_ix - h1_ix, h0_iy - h1_iy) < 40:
                    change_filter = True

            if change_filter:
                if not gesture_triggered:
                    current_filter_idx = (current_filter_idx + 1) % len(FILTERS)
                    gesture_triggered = True
            else:
                gesture_triggered = False

            # Draw glowing nodes on every fingertip
            for fx, fy in all_fingertips:
                cv2.circle(img, (fx, fy), 8, (0, 255, 255), 2)
                cv2.circle(img, (fx, fy), 4, (255, 255, 255), -1)

            # ----------------- 3. WHOLE-HAND EFFORTLESS SQUARE FRAMING -----------------
            # Automatically spans effortlessly between the extremities of both hands or all fingers
            if len(all_hand_points) >= 10:
                xs = [p[0] for p in all_hand_points]
                ys = [p[1] for p in all_hand_points]

                raw_x1 = max(0, min(xs) - 25)
                raw_y1 = max(50, min(ys) - 25)
                raw_x2 = min(w, max(xs) + 25)
                raw_y2 = min(h - 10, max(ys) + 25)

                target_box = np.array([raw_x1, raw_y1, raw_x2, raw_y2], dtype=np.float32)
                if smooth_box is None:
                    smooth_box = target_box
                else:
                    smooth_box += (target_box - smooth_box) * 0.35 # Silky smooth damping

                bx1, by1, bx2, by2 = int(smooth_box[0]), int(smooth_box[1]), int(smooth_box[2]), int(smooth_box[3])
                bw, bh = bx2 - bx1, by2 - by1

                if bw > 50 and bh > 50:
                    # ----------------- 4. RENDER LIVE PORTAL INSIDE THE FRAMED SQUARE -----------------
                    roi = img[by1:by2, bx1:bx2].copy()
                    filtered_roi = apply_filter(roi, cur_filter["id"], bx1, by1, mask_person, frame_galaxy)

                    # Smooth blend inside the square
                    img[by1:by2, bx1:bx2] = filtered_roi

                    # Glowing Neon Portal Perimeter
                    theme_col = cur_filter["theme_color"]
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), theme_col, 2)
                    cv2.rectangle(img, (bx1 - 2, by1 - 2), (bx2 + 2, by2 + 2), (255, 255, 255), 1)

                    # High-Tech Corner L-Brackets
                    bk = min(22, min(bw // 4, bh // 4))
                    for (bx, by, dx, dy) in [(bx1, by1, 1, 1), (bx2, by1, -1, 1), (bx1, by2, 1, -1), (bx2, by2, -1, -1)]:
                        cv2.line(img, (bx, by), (bx + dx * bk, by), (255, 255, 255), 3)
                        cv2.line(img, (bx, by), (bx, by + dy * bk), (255, 255, 255), 3)

                    # ----------------- 5. WHOLE-FINGER ENERGY BEAM TETHERS -----------------
                    # Connect every single finger directly to the nearest point on the portal frame!
                    for fx, fy in all_fingertips:
                        # Find closest point on rectangle perimeter
                        clamp_x = max(bx1, min(bx2, fx))
                        clamp_y = max(by1, min(by2, fy))
                        # Connect with glowing laser energy beam
                        cv2.line(img, (fx, fy), (clamp_x, clamp_y), (0, 255, 255), 2)
                        cv2.line(img, (fx, fy), (clamp_x, clamp_y), (255, 255, 255), 1)
                        # Energy spark at intersection
                        cv2.circle(img, (clamp_x, clamp_y), 4, theme_col, -1)

                    # Live Telemetry Prompt
                    portal_label = f"✨ LIVE PORTAL: {cur_filter['name']} [ PINCH TO CAPTURE ]"
                    ts = cv2.getTextSize(portal_label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)[0]
                    lx1 = max(10, (bx1 + bx2) // 2 - ts[0] // 2)
                    ly1 = max(80, by1 - 10)
                    cv2.rectangle(img, (lx1 - 4, ly1 - ts[1] - 4), (lx1 + ts[0] + 4, ly1 + 4), (16, 16, 22), -1)
                    cv2.rectangle(img, (lx1 - 4, ly1 - ts[1] - 4), (lx1 + ts[0] + 4, ly1 + 4), theme_col, 1)
                    cv2.putText(img, portal_label, (lx1, ly1), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1)

                    # ----------------- 6. PINCH TO CAPTURE FROZEN PHOTO IN FRONT -----------------
                    now = time.time()
                    if pinch_detected and (now - last_capture_time > 0.7):
                        # Capture exact frozen image snapshot of that scenario
                        new_pic = CapturedPortalPicture(bx1, by1, bx2, by2, cur_filter, filtered_roi)
                        captured_pictures.append(new_pic) # Embedded in front!
                        last_capture_time = now
                        flash_timer = 3

                        # Automatically rotate to next filter for the next square capture!
                        current_filter_idx = (current_filter_idx + 1) % len(FILTERS)
                        print(f"📸 CAPTURED FROZEN PICTURE #{len(captured_pictures)} [{cur_filter['name']}] IN FRONT! Next: [{FILTERS[current_filter_idx]['name']}]")
        else:
            smooth_box = None

        # Camera Shutter Flash
        if flash_timer > 0:
            flash_overlay = np.full_like(img, 255)
            cv2.addWeighted(flash_overlay, 0.45, img, 0.55, 0, img)
            flash_timer -= 1

        # ----------------- TOP HUD & TELEMETRY -----------------
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), (14, 14, 20), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cur_theme = cur_filter["theme_color"]
        cv2.line(img, (0, 48), (w, 48), cur_theme, 2)

        hud_1 = f"FILTER: {cur_filter['name']}  |  PHOTOS: {len(captured_pictures)}"
        hud_2 = "🖐️ Fingers connect with portal  |  👌 PINCH = Capture photo in front  |  🤙 Jempol+Kelingking = Switch  |  'c' = Clear"
        cv2.putText(img, hud_1, (15, 20), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 255, 255), 1)
        cv2.putText(img, hud_2, (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        cv2.imshow('RetroLens - Full-Finger Connected Portal & Photo Capture', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            captured_pictures.clear()
            print("🗑️ All photos cleared.")
        elif key == ord('n'):
            current_filter_idx = (current_filter_idx + 1) % len(FILTERS)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()