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

# ----------------- RICH EXPANDED DIMENSIONAL FILTERS -----------------
DIMENSIONS = [
    {"name": "COSMIC GALAXY 🌌", "id": "GALAXY", "theme_color": (255, 200, 0)},
    {"name": "TOUCHDESIGNER POP 🔮", "id": "POP_PLEXUS", "theme_color": (203, 19, 255)},
    {"name": "CYBERPUNK MATRIX ⚡", "id": "MATRIX", "theme_color": (0, 255, 128)},
    {"name": "KALEIDOSCOPE WARP 🌈", "id": "KALEIDO", "theme_color": (255, 0, 200)},
    {"name": "PREDATOR THERMAL 🧊", "id": "THERMAL", "theme_color": (0, 140, 255)},
    {"name": "80s SYNTHWAVE 👾", "id": "SYNTHWAVE", "theme_color": (255, 60, 180)},
    {"name": "NEON WIREFRAME 💎", "id": "NEON", "theme_color": (255, 255, 0)},
    {"name": "DUAL-TONE COMIC 🎭", "id": "DUALTONE", "theme_color": (0, 165, 255)},
    {"name": "RGB GLITCH 📺", "id": "GLITCH", "theme_color": (255, 0, 100)},
    {"name": "PENCIL SKETCH ✏️", "id": "SKETCH", "theme_color": (200, 200, 200)},
    {"name": "INVERTED X-RAY 👁️", "id": "INVERT", "theme_color": (0, 255, 255)},
    {"name": "8-BIT PIXEL ART 🎮", "id": "PIXEL", "theme_color": (50, 255, 50)},
    {"name": "WARM SEPIA FILM 🎞️", "id": "SEPIA", "theme_color": (0, 180, 220)}
]

# MediaPipe Hand Skeleton Connection Map (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm cross-connections
]


# ----------------- VISUAL SHOCKWAVE -----------------
class Shockwave:
    def __init__(self, x, y, color=(0, 255, 255), max_radius=80):
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

    def add_shockwave(self, x, y, color=(0, 255, 255), max_radius=80):
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


# ----------------- CAPTURED PORTAL SQUARE WINDOW OBJECT -----------------
class CapturedPortalSquare:
    """Represents a permanently captured spatial square window that retains its own unique dimension effect."""
    def __init__(self, x1, y1, x2, y2, dimension_dict, frame_snapshot=None):
        self.x1 = int(min(x1, x2))
        self.y1 = int(min(y1, y2))
        self.x2 = int(max(x1, x2))
        self.y2 = int(max(y1, y2))
        self.w = max(40, self.x2 - self.x1)
        self.h = max(40, self.y2 - self.y1)
        self.dim = dimension_dict
        self.created_at = time.time()
        self.frozen_snapshot = frame_snapshot.copy() if frame_snapshot is not None else None
        self.is_live = True # Render live stream in this dimension or freeze frame
        self.border_phase = random.uniform(0, math.pi * 2)
        self.id_num = random.randint(100, 999)

    def contains(self, px, py):
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def move_to(self, cx, cy, screen_w, screen_h):
        hw = self.w // 2
        hh = self.h // 2
        self.x1 = max(0, min(screen_w - self.w, cx - hw))
        self.y1 = max(60, min(screen_h - self.h - 35, cy - hh))
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    def draw_window(self, img, full_dim_frame, particles):
        h, w = img.shape[:2]
        sx1, sx2 = max(0, self.x1), min(w, self.x2)
        sy1, sy2 = max(0, self.y1), min(h, self.y2)
        if sx2 <= sx1 or sy2 <= sy1:
            return

        # 1. Composite the Dimension Effect inside this Square Boundary
        if self.is_live:
            img[sy1:sy2, sx1:sx2] = full_dim_frame[sy1:sy2, sx1:sx2]
        elif self.frozen_snapshot is not None:
            frozen_crop = cv2.resize(self.frozen_snapshot, (self.w, self.h))
            img[sy1:sy2, sx1:sx2] = frozen_crop[:sy2-sy1, :sx2-sx1]

        # 2. Glowing Event Horizon Neon Border
        col = self.dim["theme_color"]
        t = time.time()
        pulse = 0.85 + 0.15 * math.sin(t * 4.0 + self.border_phase)
        border_col = (int(col[0] * pulse), int(col[1] * pulse), int(col[2] * pulse))

        cv2.rectangle(img, (sx1, sy1), (sx2, sy2), border_col, 2)
        cv2.rectangle(img, (sx1 + 2, sy1 + 2), (sx2 - 2, sy2 - 2), (255, 255, 255), 1)

        # 3. High-Tech Corner Framing Brackets
        bk = min(18, min(self.w // 4, self.h // 4))
        for (bx, by, dx, dy) in [(sx1, sy1, 1, 1), (sx2, sy1, -1, 1), (sx1, sy2, 1, -1), (sx2, sy2, -1, -1)]:
            cv2.line(img, (bx, by), (bx + dx * bk, by), (255, 255, 255), 3)
            cv2.line(img, (bx, by), (bx, by + dy * bk), (255, 255, 255), 3)

        # 4. Hologram Dimension Label Badge
        badge_text = f"🪟 {self.dim['name'].split()[0]}"
        ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        badge_x = sx1 + 4
        badge_y = max(75, sy1 - 8)
        cv2.rectangle(img, (badge_x - 3, badge_y - ts[1] - 4), (badge_x + ts[0] + 6, badge_y + 4), (18, 18, 26), -1)
        cv2.rectangle(img, (badge_x - 3, badge_y - ts[1] - 4), (badge_x + ts[0] + 6, badge_y + 4), border_col, 1)
        cv2.putText(img, badge_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)


# ----------------- DIMENSIONAL EFFECT GENERATOR -----------------
class DimensionalRenderer:
    def __init__(self, w=1280, h=720):
        self.w = w
        self.h = h
        self.galaxy_bg = np.zeros((h, w, 3), dtype=np.uint8)
        self.galaxy_bg[:] = (24, 8, 36)
        for _ in range(700):
            sx = np.random.randint(0, w)
            sy = np.random.randint(0, h)
            self.galaxy_bg[sy, sx] = (255, 255, 255)
        for _ in range(80):
            sx = np.random.randint(0, w)
            sy = np.random.randint(0, h)
            cv2.circle(self.galaxy_bg, (sx, sy), np.random.randint(2, 5), (np.random.randint(160, 255), np.random.randint(100, 255), 255), -1)

    def render_dimension(self, frame, dim_id, mask_person=None):
        h, w = frame.shape[:2]
        t = time.time()

        # 1. 🌌 COSMIC GALAXY
        if dim_id == "GALAXY":
            if self.galaxy_bg.shape[:2] != (h, w):
                self.galaxy_bg = cv2.resize(self.galaxy_bg, (w, h))
            dim_frame = self.galaxy_bg.copy()
            galaxy_center = (w // 2, h // 2)
            for arm in range(3):
                arm_angle = t * 0.8 + arm * (2 * math.pi / 3.0)
                for r_step in range(15, 220, 10):
                    ang = arm_angle + (r_step * 0.035)
                    gx = int(galaxy_center[0] + math.cos(ang) * r_step)
                    gy = int(galaxy_center[1] + math.sin(ang) * r_step)
                    if 0 <= gx < w and 0 <= gy < h:
                        cv2.circle(dim_frame, (gx, gy), random.randint(2, 5), (255, random.randint(180, 240), random.randint(0, 100)), -1)
            if mask_person is not None:
                if mask_person.shape[:2] != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_3c = cv2.cvtColor(mask_person, cv2.COLOR_GRAY2BGR) / 255.0
                dim_frame = dim_frame * (1.0 - mask_3c * 0.35) + frame * (mask_3c * 0.35)
            return np.clip(dim_frame, 0, 255).astype(np.uint8)

        # 2. 🔮 TOUCHDESIGNER PROXIMITY POP
        elif dim_id == "POP_PLEXUS":
            dim_frame = np.zeros((h, w, 3), dtype=np.uint8)
            stride = 14
            nodes = []
            if mask_person is not None:
                y_indices, x_indices = np.where(mask_person > 0)
                if len(x_indices) > 0:
                    min_x, max_x = np.min(x_indices), np.max(x_indices)
                    min_y, max_y = np.min(y_indices), np.max(y_indices)
                    for gy in range(min_y, max_y, stride):
                        for gx in range(min_x, max_x, stride):
                            if 0 <= gy < h and 0 <= gx < w and mask_person[gy, gx] > 0:
                                px = int(gx + math.sin(gy * 0.04 + t * 3.0) * 3.0)
                                py = int(gy + math.cos(gx * 0.04 + t * 3.0) * 3.0)
                                col = (203, 19, 255) if py < h // 2 else (245, 248, 255)
                                nodes.append((px, py, col))
            cell_size = 28
            grid = {}
            for idx, (nx, ny, col) in enumerate(nodes):
                key = (nx // cell_size, ny // cell_size)
                if key not in grid: grid[key] = []
                grid[key].append(idx)
            for idx, (x1, y1, col1) in enumerate(nodes):
                cx_c, cy_c = x1 // cell_size, y1 // cell_size
                for gx in range(cx_c - 1, cx_c + 2):
                    for gy in range(cy_c - 1, cy_c + 2):
                        if (gx, gy) in grid:
                            for n_idx in grid[(gx, gy)]:
                                if n_idx <= idx: continue
                                x2, y2, col2 = nodes[n_idx]
                                if math.hypot(x2 - x1, y2 - y1) < 25.0:
                                    cv2.line(dim_frame, (x1, y1), (x2, y2), (200, 200, 255), 1)
            for x, y, col in nodes:
                cv2.circle(dim_frame, (x, y), 2, col, -1)
            return dim_frame

        # 3. ⚡ CYBERPUNK MATRIX
        elif dim_id == "MATRIX":
            dim_frame = np.zeros((h, w, 3), dtype=np.uint8)
            horizon_y = h // 2
            for gy in range(horizon_y, h, 20):
                cv2.line(dim_frame, (0, gy), (w, gy), (0, 255, 128), 1)
            for gx in range(0, w, 35):
                cv2.line(dim_frame, (w // 2, horizon_y), (gx, h), (0, 255, 128), 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
            edges_bgr = np.zeros_like(frame)
            edges_bgr[edges > 0] = (0, 255, 128)
            return cv2.add(dim_frame, edges_bgr)

        # 4. 🌈 KALEIDOSCOPE WARP
        elif dim_id == "KALEIDO":
            quad = frame[:h//2, :w//2]
            quad_flip_h = cv2.flip(quad, 1)
            top_half = np.hstack((quad, quad_flip_h))
            bottom_half = cv2.flip(top_half, 0)
            kaleido = np.vstack((top_half, bottom_half))
            if kaleido.shape[:2] != (h, w):
                kaleido = cv2.resize(kaleido, (w, h))
            hsv = cv2.cvtColor(kaleido, cv2.COLOR_BGR2HSV)
            shift_hue = int((t * 40) % 180)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int32) + shift_hue) % 180
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5. 🧊 PREDATOR THERMAL
        elif dim_id == "THERMAL":
            return cv2.applyColorMap(frame, cv2.COLORMAP_JET)

        # 6. 👾 80s SYNTHWAVE
        elif dim_id == "SYNTHWAVE":
            synth = frame.copy()
            synth[:, :, 0] = np.clip(synth[:, :, 0] * 1.4, 0, 255)
            synth[:, :, 2] = np.clip(synth[:, :, 2] * 1.5, 0, 255)
            synth[:, :, 1] = np.clip(synth[:, :, 1] * 0.4, 0, 255)
            for sy in range(0, h, 6):
                synth[sy, :] = synth[sy, :] // 2
            return synth

        # 7. 💎 NEON WIREFRAME
        elif dim_id == "NEON":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            edges_bgr = np.zeros_like(frame)
            edges_bgr[edges > 0] = (255, 255, 0)
            kernel = np.ones((3, 3), np.uint8)
            return cv2.dilate(edges_bgr, kernel, iterations=1)

        # 8. 🎭 DUAL-TONE COMIC
        elif dim_id == "DUALTONE":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            dual = np.zeros_like(frame)
            dual[mask_c == 255] = [0, 165, 255]
            dual[mask_c == 0] = [203, 19, 255]
            return dual

        # 9. 📺 RGB GLITCH
        elif dim_id == "GLITCH":
            glitch = frame.copy()
            shift = max(8, w // 30)
            if w > shift:
                glitch[:, :-shift, 2] = frame[:, shift:, 2]
                glitch[:, shift:, 0] = frame[:, :-shift, 0]
            return glitch

        # 10. ✏️ PENCIL SKETCH
        elif dim_id == "SKETCH":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            inv = cv2.bitwise_not(gray)
            blur = cv2.GaussianBlur(inv, (21, 21), 0)
            sketch = cv2.divide(gray, 255 - blur, scale=256)
            return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

        # 11. 👁️ INVERTED X-RAY
        elif dim_id == "INVERT":
            return cv2.bitwise_not(frame)

        # 12. 🎮 8-BIT PIXEL ART
        elif dim_id == "PIXEL":
            small = cv2.resize(frame, (max(1, w // 16), max(1, h // 16)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 13. 🎞️ WARM SEPIA FILM
        elif dim_id == "SEPIA":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            sepia = cv2.transform(frame, kernel)
            return np.clip(sepia, 0, 255).astype(np.uint8)

        return frame


# ----------------- MULTI-PORTAL SPATIAL CANVAS ENGINE -----------------
class SpatialPortalMatrix:
    """Manages two-hand square framing, pinch-to-capture dimensional windows, and multi-portal rendering."""
    def __init__(self, w=1280, h=720):
        self.w = w
        self.h = h
        self.renderer = DimensionalRenderer(w, h)
        self.captured_windows = [] # List of CapturedPortalSquare
        self.next_dim_index = 0    # Cycles automatically to next dimension on each capture
        self.last_capture_time = 0.0
        self.cached_dim_frames = {}
        self.last_cache_time = 0.0

        # Current Live Two-Hand Framing Box
        self.framing_active = False
        self.frame_box = (0, 0, 0, 0)
        self.frame_corner1 = (0, 0)
        self.frame_corner2 = (0, 0)

        # Active Dragged Window
        self.dragged_window = None

    def update_hand_framing(self, hand_landmarks, w, h, pinching_flags, particles):
        now = time.time()
        self.framing_active = False

        if hand_landmarks and len(hand_landmarks) >= 2:
            # Two Hands Available -> Frame Square Area
            h1 = hand_landmarks[0]
            h2 = hand_landmarks[1]

            # Index Fingertips define opposite corners of the square
            p1_x, p1_y = int(h1[8].x * w), int(h1[8].y * h)
            p2_x, p2_y = int(h2[8].x * w), int(h2[8].y * h)

            self.frame_corner1 = (p1_x, p1_y)
            self.frame_corner2 = (p2_x, p2_y)

            x1 = min(p1_x, p2_x)
            y1 = min(p1_y, p2_y)
            x2 = max(p1_x, p2_x)
            y2 = max(p1_y, p2_y)

            # Ensure minimum size
            if (x2 - x1) >= 45 and (y2 - y1) >= 45:
                self.framing_active = True
                self.frame_box = (x1, y1, x2, y2)

                # ----------------- PINCH TO CAPTURE SQUARE PORTAL -----------------
                is_pinching = any(pinching_flags)
                if is_pinching and (now - self.last_capture_time > 0.8):
                    # CAPTURE THIS SQUARE WITH CURRENT DIMENSION!
                    cur_dim = DIMENSIONS[self.next_dim_index]
                    new_portal = CapturedPortalSquare(x1, y1, x2, y2, cur_dim)
                    self.captured_windows.append(new_portal)

                    # Trigger visual effects
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    particles.emit_fireworks(cx, cy, count=35, color=cur_dim["theme_color"])
                    particles.add_shockwave(cx, cy, cur_dim["theme_color"], max_radius=100)

                    # Cycle to the NEXT dimension automatically for the next square capture!
                    self.next_dim_index = (self.next_dim_index + 1) % len(DIMENSIONS)
                    self.last_capture_time = now
                    print(f"✨ CAPTURED SQUARE PORTAL #{len(self.captured_windows)}: {cur_dim['name']}")

        elif hand_landmarks and len(hand_landmarks) == 1:
            # Single Hand Drag / Interaction
            h1 = hand_landmarks[0]
            ix = int(h1[8].x * w)
            iy = int(h1[8].y * h)
            is_pinch = pinching_flags[0] if pinching_flags else False

            if is_pinch:
                if self.dragged_window is None:
                    # Check if pinching inside any existing window
                    for win in reversed(self.captured_windows):
                        if win.contains(ix, iy):
                            self.dragged_window = win
                            break
                else:
                    self.dragged_window.move_to(ix, iy, w, h)
            else:
                self.dragged_window = None

    def clear_all_portals(self, particles=None):
        if self.captured_windows:
            self.captured_windows.clear()
            if particles:
                particles.emit_fireworks(self.w // 2, self.h // 2, count=40)

    def undo_last_portal(self, particles=None):
        if self.captured_windows:
            popped = self.captured_windows.pop()
            if particles:
                cx = (popped.x1 + popped.x2) // 2
                cy = (popped.y1 + popped.y2) // 2
                particles.add_shockwave(cx, cy, (0, 255, 255), max_radius=80)

    def render_canvas(self, frame, mask_person, particles):
        h, w = frame.shape[:2]
        t = time.time()

        # 1. Update Dimension Renderers for all Active Dimensions
        active_dim_ids = set([win.dim["id"] for win in self.captured_windows])
        if self.framing_active:
            active_dim_ids.add(DIMENSIONS[self.next_dim_index]["id"])

        rendered_dims = {}
        for dim_id in active_dim_ids:
            rendered_dims[dim_id] = self.renderer.render_dimension(frame, dim_id, mask_person)

        # 2. Draw All Captured Spatial Windows (Each retains its own unique dimension effect!)
        for win in self.captured_windows:
            dim_id = win.dim["id"]
            if dim_id in rendered_dims:
                win.draw_window(frame, rendered_dims[dim_id], particles)

        # 3. Draw Live Two-Hand Hologram Framing Preview Box
        if self.framing_active:
            x1, y1, x2, y2 = self.frame_box
            candidate_dim = DIMENSIONS[self.next_dim_index]
            theme_col = candidate_dim["theme_color"]

            # Live preview of candidate dimension inside the framing box
            dim_preview = rendered_dims[candidate_dim["id"]]
            overlay_crop = dim_preview[y1:y2, x1:x2]
            alpha_blend = 0.82
            frame[y1:y2, x1:x2] = cv2.addWeighted(overlay_crop, alpha_blend, frame[y1:y2, x1:x2], 1.0 - alpha_blend, 0)

            # Animated laser framing border
            cv2.rectangle(frame, (x1, y1), (x2, y2), theme_col, 2)
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

            # Hand Corner Markers
            c1x, c1y = self.frame_corner1
            c2x, c2y = self.frame_corner2
            cv2.drawMarker(frame, (c1x, c1y), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.drawMarker(frame, (c2x, c2y), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.line(frame, (c1x, c1y), (c2x, c1y), (0, 255, 255), 1)
            cv2.line(frame, (c1x, c1y), (c1x, c2y), (0, 255, 255), 1)
            cv2.line(frame, (c2x, c2y), (c2x, c1y), (0, 255, 255), 1)
            cv2.line(frame, (c2x, c2y), (c1x, c2y), (0, 255, 255), 1)

            # Telemetry Prompt
            hud_msg = f"📸 PINCH TO CAPTURE [{candidate_dim['name']}]"
            ts = cv2.getTextSize(hud_msg, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)[0]
            hud_x = max(10, (x1 + x2) // 2 - ts[0] // 2)
            hud_y = max(85, y1 - 10)
            cv2.rectangle(frame, (hud_x - 6, hud_y - ts[1] - 6), (hud_x + ts[0] + 6, hud_y + 6), (18, 18, 26), -1)
            cv2.rectangle(frame, (hud_x - 6, hud_y - ts[1] - 6), (hud_x + ts[0] + 6, hud_y + 6), theme_col, 1)
            cv2.putText(frame, hud_msg, (hud_x, hud_y), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)


# ----------------- ULTRA-RESPONSIVE SPATIAL BUTTONS -----------------
class SpatialButton:
    def __init__(self, bid, x, y, w, h, text, color=(28, 28, 38), border_color=(0, 255, 255), hold_time=0.30):
        self.bid = bid
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.text = text
        self.color = color
        self.border_color = border_color
        self.hold_time = hold_time
        self.hover_start = None
        self.hover_progress = 0.0
        self.just_clicked = 0

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def update_hover(self, is_hovering, is_pinching=False):
        now = time.time()
        if is_hovering:
            if is_pinching:
                self.hover_start = None
                self.hover_progress = 0.0
                self.just_clicked = 5
                return True

            if self.hover_start is None:
                self.hover_start = now
            elapsed = now - self.hover_start
            self.hover_progress = min(1.0, elapsed / self.hold_time)
            if elapsed >= self.hold_time:
                self.hover_start = None
                self.hover_progress = 0.0
                self.just_clicked = 5
                return True
        else:
            self.hover_start = None
            self.hover_progress = 0.0
        return False

    def draw(self, img, is_active=False):
        if self.just_clicked > 0:
            bg_col = (0, 200, 100)
            self.just_clicked -= 1
        elif self.hover_progress > 0:
            bg_col = (55, 55, 80)
        elif is_active:
            bg_col = (45, 30, 60)
        else:
            bg_col = self.color

        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_col, -1)
        border_col = (0, 255, 0) if is_active else self.border_color
        thickness = 2 if is_active or self.hover_progress > 0 else 1
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border_col, thickness)

        if self.hover_progress > 0:
            fill_w = int(self.w * self.hover_progress)
            cv2.rectangle(img, (self.x, self.y + self.h - 4), (self.x + fill_w, self.y + self.h), (0, 255, 0), -1)

        ts = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        tx = self.x + (self.w - ts[0]) // 2
        ty = self.y + (self.h + ts[1]) // 2
        txt_col = (255, 255, 255) if not is_active else (0, 255, 255)
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1)


# ----------------- GESTURE DETECTOR -----------------
class GestureRecognizer:
    @classmethod
    def classify(cls, landmarks, w, h):
        tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)
        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
        pinch_dist = math.hypot(tx - ix, ty - iy)
        if pinch_dist < 45:
            return "PINCH"

        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"

        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM"

        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST"

        return "UNKNOWN"


# ----------------- SKELETON RENDERING -----------------
def draw_hand_skeleton(img, hand_lms, w, h, particles, hand_idx=0):
    pts = []
    for i in range(21):
        px = int(hand_lms[i].x * w)
        py = int(hand_lms[i].y * h)
        pts.append((px, py))

    for c1, c2 in HAND_CONNECTIONS:
        p1, p2 = pts[c1], pts[c2]
        cv2.line(img, p1, p2, (0, 80, 80), 4)
        cv2.line(img, p1, p2, (0, 255, 255), 2)

    for i in [4, 8, 12, 16, 20]:
        cv2.circle(img, pts[i], 7, (0, 255, 255), 2)
        cv2.circle(img, pts[i], 4, (255, 255, 255), -1)


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
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

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

    print("✨ MULTI-PORTAL SPATIAL SQUARE CANVAS STUDIO ✨")

    last_timestamp_ms = 0
    flash_timer = 0
    fps_time = time.time()
    fps = 30.0
    exit_requested = False

    particles = ParticleManager()
    portal_matrix = SpatialPortalMatrix()

    while not exit_requested:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        if portal_matrix.w != w or portal_matrix.h != h:
            portal_matrix.w, portal_matrix.h = w, h

        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        mask_person = None
        if segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

        pointers = []
        pinching_flags = []

        if hand_results.hand_landmarks:
            for h_idx, hand_lms in enumerate(hand_results.hand_landmarks):
                g_type = GestureRecognizer.classify(hand_lms, w, h)
                index_pt = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                pointers.append(index_pt)
                pinching_flags.append(g_type == "PINCH")

                # Skeleton Visualization
                draw_hand_skeleton(img, hand_lms, w, h, particles, hand_idx=h_idx)

        # ----------------- 1. TWO-HAND SQUARE FRAMING & CAPTURE LOGIC -----------------
        portal_matrix.update_hand_framing(
            hand_results.hand_landmarks if hand_results else None,
            w, h,
            pinching_flags,
            particles
        )

        # ----------------- 2. RENDER MULTI-PORTAL CANVAS -----------------
        portal_matrix.render_canvas(img, mask_person, particles)

        # ----------------- 3. TOUCHLESS SPATIAL BUTTONS -----------------
        buttons = []
        next_dim_info = DIMENSIONS[portal_matrix.next_dim_index]
        buttons.append(SpatialButton("NEXT_DIM", 10, 10, 150, 38, f"NEXT: {next_dim_info['name'].split()[0]} ▶", border_color=next_dim_info["theme_color"]))
        buttons.append(SpatialButton("UNDO", 166, 10, 78, 38, "↩ UNDO", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("CLEAR_ALL", 250, 10, 96, 38, "🗑️ CLEAR ALL", border_color=(255, 0, 100)))

        # Dimension Shortcut Swatches
        dim_start_x = 354
        for d_i in range(min(5, len(DIMENSIONS))):
            d_item = DIMENSIONS[d_i]
            d_short = d_item["name"].split()[0][:3]
            buttons.append(SpatialButton(f"SET_DIM_{d_i}", dim_start_x + (d_i * 38), 10, 34, 38, d_short, border_color=d_item["theme_color"]))

        # Universal Action Buttons
        buttons.append(SpatialButton("PHOTO", w - 158, 10, 76, 38, "📸 SNAP", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("EXIT", w - 80, 10, 72, 38, "❌ EXIT", border_color=(0, 0, 255), hold_time=0.9))

        # ----------------- BUTTON HOVER & TRIGGER HANDLING -----------------
        for btn in buttons:
            is_hover = False
            is_pinch = False
            for p_idx, pt in enumerate(pointers):
                if btn.contains(pt[0], pt[1]):
                    is_hover = True
                    is_pinch = pinching_flags[p_idx]
                    break

            triggered = btn.update_hover(is_hover, is_pinch)
            if triggered:
                particles.add_shockwave(btn.x + btn.w // 2, btn.y + btn.h // 2, (0, 255, 255), max_radius=50)
                if btn.bid == "NEXT_DIM":
                    portal_matrix.next_dim_index = (portal_matrix.next_dim_index + 1) % len(DIMENSIONS)
                elif btn.bid == "UNDO":
                    portal_matrix.undo_last_portal(particles)
                elif btn.bid == "CLEAR_ALL":
                    portal_matrix.clear_all_portals(particles)
                elif btn.bid.startswith("SET_DIM_"):
                    d_idx = int(btn.bid.split("_")[2])
                    portal_matrix.next_dim_index = d_idx
                elif btn.bid == "PHOTO":
                    fname = f"multi_portal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fpath = os.path.join(SCREENSHOTS_DIR, fname)
                    cv2.imwrite(fpath, img)
                    print(f"📸 Photo saved: {fpath}")
                    flash_timer = 4
                elif btn.bid == "EXIT":
                    exit_requested = True

        # Draw Interactive Cursor Reticles at Index Fingertips
        for p_idx, pt in enumerate(pointers):
            cv2.circle(img, pt, 14, (0, 255, 255), 1)
            cv2.circle(img, pt, 4, (255, 255, 255), -1)
            cv2.line(img, (pt[0] - 18, pt[1]), (pt[0] - 8, pt[1]), (0, 255, 255), 1)
            cv2.line(img, (pt[0] + 8, pt[1]), (pt[0] + 18, pt[1]), (0, 255, 255), 1)
            cv2.line(img, (pt[0], pt[1] - 18), (pt[0], pt[1] - 8), (0, 255, 255), 1)
            cv2.line(img, (pt[0], pt[1] + 8), (pt[0], pt[1] + 18), (0, 255, 255), 1)

            for btn in buttons:
                if btn.contains(pt[0], pt[1]) and btn.hover_progress > 0:
                    sweep_angle = int(360 * btn.hover_progress)
                    cv2.ellipse(img, pt, (22, 22), 0, -90, -90 + sweep_angle, (0, 255, 0), 3)
                    break

        particles.update_and_draw(img)

        # Top HUD Bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (16, 16, 24), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cur_next_col = DIMENSIONS[portal_matrix.next_dim_index]["theme_color"]
        cv2.line(img, (0, 56), (w, 56), cur_next_col, 2)

        for btn in buttons:
            btn.draw(img)

        # Bottom Bar & Telemetry
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        portal_count = len(portal_matrix.captured_windows)
        status_text = f"🖼️ CAPTURED PORTALS: {portal_count}  |  🖐️ Frame with 2 hands + 👌 Pinch to CAPTURE square  |  Each square retains its effect & changes every capture!"
        cv2.putText(img, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)

        now = time.time()
        fps = 1.0 / max(0.001, now - fps_time)
        fps_time = now
        cv2.putText(img, f"FPS: {int(fps)}", (w - 85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if flash_timer > 0:
            cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
            flash_timer -= 1

        cv2.imshow('Multi-Portal Spatial Canvas - Frame, Pinch-Capture & Retain Dimensions', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()