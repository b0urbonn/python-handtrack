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

# ----------------- PORTAL CONFIG & DIMENSIONS -----------------
DIMENSIONS = [
    {"name": "COSMIC GALAXY 🌌", "id": "GALAXY", "theme_color": (255, 200, 0)},
    {"name": "TOUCHDESIGNER POP 🔮", "id": "POP_PLEXUS", "theme_color": (203, 19, 255)},
    {"name": "CYBERPUNK MATRIX ⚡", "id": "MATRIX", "theme_color": (0, 255, 128)},
    {"name": "KALEIDOSCOPE WARP 🌈", "id": "KALEIDO", "theme_color": (255, 0, 200)},
    {"name": "PREDATOR THERMAL 🧊", "id": "THERMAL", "theme_color": (0, 140, 255)},
    {"name": "80s SYNTHWAVE 👾", "id": "SYNTHWAVE", "theme_color": (255, 60, 180)}
]

PORTAL_SHAPES = ["SPARK RING (DR STRANGE)", "DIMENSIONAL OVAL", "POLYGONAL FRAME"]

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
    def __init__(self, x, y, color=(0, 255, 255), max_radius=75):
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


# ----------------- PORTAL PARTICLE ENGINE -----------------
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
        elif self.p_type == "portal_spark":
            self.vx += random.uniform(-0.5, 0.5)
            self.vy += random.uniform(-0.5, 0.5)
        self.life -= 1.0
        self.size = max(0.5, self.size * 0.95)
        return self.life > 0

    def draw(self, img):
        if self.life <= 0:
            return
        alpha = min(1.0, max(0.0, self.life / self.max_life))
        c = self.color
        if isinstance(c, tuple):
            color = (int(c[0] * alpha), int(c[1] * alpha), int(c[2] * alpha))
        else:
            color = (255, 255, 255)
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

    def add_shockwave(self, x, y, color=(0, 255, 255), max_radius=75):
        self.shockwaves.append(Shockwave(x, y, color, max_radius))

    def emit_sparkles(self, x, y, color=(0, 255, 255), count=3, speed=3):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(1, speed)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, color, random.uniform(2, 5), random.randint(15, 30)))

    def emit_portal_rim_sparks(self, cx, cy, rx, ry, color=(0, 200, 255), count=4):
        """Emits Doctor Strange-style fiery sparks swirling along the portal event horizon rim."""
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            px = cx + math.cos(angle) * rx + random.uniform(-6, 6)
            py = cy + math.sin(angle) * ry + random.uniform(-6, 6)
            tangent = angle + math.pi / 2.0
            spd = random.uniform(2.5, 7.0)
            vx = math.cos(tangent) * spd + random.uniform(-1, 1)
            vy = math.sin(tangent) * spd + random.uniform(-1, 1)
            spark_colors = [color, (0, 165, 255), (0, 230, 255), (255, 255, 255)]
            c = random.choice(spark_colors)
            self.add_particle(Particle(px, py, vx, vy, c, random.uniform(2, 5), random.randint(10, 22), decay=0.94, p_type="portal_spark"))

    def emit_fireworks(self, x, y, count=35):
        colors = [(0, 165, 255), (255, 255, 0), (255, 50, 50), (50, 255, 50), (255, 0, 255), (0, 255, 255)]
        for _ in range(count):
            c = random.choice(colors)
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(3, 11)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, c, random.uniform(3, 6), random.randint(20, 45), decay=0.98, p_type="firework"))
        self.add_shockwave(x, y, (255, 255, 255), max_radius=85)

    def update_and_draw(self, img):
        self.particles = [p for p in self.particles if p.update() and (p.draw(img) or True)]
        self.shockwaves = [sw for sw in self.shockwaves if sw.update() and (sw.draw(img) or True)]


# ----------------- HYPER-INTERACTIVE QUANTUM PORTAL ENGINE -----------------
class QuantumPortalEngine:
    """Manages the dimensional gateway, interactive two-hand framing, event horizon rim, and 6 alternate universes."""
    
    def __init__(self, w=1280, h=720):
        self.w = w
        self.h = h
        self.dim_idx = 0         # Active Dimension Index
        self.shape_idx = 0       # 0: SPARK RING, 1: OVAL, 2: POLYGON
        
        # Spatial Portal Geometry
        self.cx = float(w // 2)
        self.cy = float(h // 2)
        self.rx = 160.0
        self.ry = 160.0
        self.target_cx = float(w // 2)
        self.target_cy = float(h // 2)
        self.target_rx = 160.0
        self.target_ry = 160.0
        self.rotation_angle = 0.0
        self.rim_spin = 0.0
        self.open_strength = 1.0
        
        # Galaxy Dimension Asset Cache
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

    def next_dimension(self, particles=None):
        self.dim_idx = (self.dim_idx + 1) % len(DIMENSIONS)
        dim = DIMENSIONS[self.dim_idx]
        if particles:
            particles.add_shockwave(int(self.cx), int(self.cy), dim["theme_color"], max_radius=110)
            particles.emit_fireworks(int(self.cx), int(self.cy), count=30)

    def next_shape(self, particles=None):
        self.shape_idx = (self.shape_idx + 1) % len(PORTAL_SHAPES)
        if particles:
            particles.add_shockwave(int(self.cx), int(self.cy), (0, 255, 255), max_radius=80)

    def set_dimension_by_index(self, idx, particles=None):
        if 0 <= idx < len(DIMENSIONS) and idx != self.dim_idx:
            self.dim_idx = idx
            dim = DIMENSIONS[self.dim_idx]
            if particles:
                particles.add_shockwave(int(self.cx), int(self.cy), dim["theme_color"], max_radius=110)

    def update_geometry(self, hand_landmarks, w, h, particles):
        self.rim_spin += 0.08
        t = time.time()

        if hand_landmarks and len(hand_landmarks) >= 2:
            # ----------------- TWO-HAND INTERACTIVE PORTAL FRAMING -----------------
            h1 = hand_landmarks[0]
            h2 = hand_landmarks[1]
            
            # Use Index Fingertips and Thumbs for Precision Spatial Aperture
            h1_x, h1_y = int(h1[8].x * w), int(h1[8].y * h)
            h2_x, h2_y = int(h2[8].x * w), int(h2[8].y * h)

            self.target_cx = (h1_x + h2_x) / 2.0
            self.target_cy = (h1_y + h2_y) / 2.0

            dx = h2_x - h1_x
            dy = h2_y - h1_y
            hand_dist = math.hypot(dx, dy)
            self.rotation_angle = math.degrees(math.atan2(dy, dx))

            # Dynamic Portal Aperture Radius based on Hand Distance
            self.target_rx = max(50.0, min(380.0, hand_dist * 0.70))
            self.target_ry = max(50.0, min(340.0, hand_dist * 0.65))
            self.open_strength = min(1.0, hand_dist / 140.0)

        elif hand_landmarks and len(hand_landmarks) == 1:
            # ----------------- ONE-HAND ORBITAL PORTAL -----------------
            h1 = hand_landmarks[0]
            palm_x = int(h1[9].x * w)
            palm_y = int(h1[9].y * h)

            self.target_cx = palm_x
            self.target_cy = max(70, palm_y - 120)
            self.target_rx = 135.0
            self.target_ry = 135.0
            self.rotation_angle = 0.0
            self.open_strength = 1.0
        else:
            # Gentle Floating Rest State in Screen Center
            self.target_cx = w // 2 + math.sin(t * 1.5) * 45.0
            self.target_cy = h // 2 + math.cos(t * 1.2) * 30.0
            self.target_rx = 155.0 + math.sin(t * 2.0) * 15.0
            self.target_ry = 155.0 + math.cos(t * 2.0) * 15.0
            self.rotation_angle = t * 10.0

        # Smooth Elastic Interpolation
        self.cx += (self.target_cx - self.cx) * 0.25
        self.cy += (self.target_cy - self.cy) * 0.25
        self.rx += (self.target_rx - self.rx) * 0.22
        self.ry += (self.target_ry - self.ry) * 0.22

    def render_dimension_content(self, frame, mask_person, hand_landmarks, face_landmarks):
        """Generates the full alternate universe image to be revealed inside the portal."""
        h, w = frame.shape[:2]
        dim = DIMENSIONS[self.dim_idx]
        dim_id = dim["id"]
        t = time.time()

        # 1. 🌌 COSMIC GALAXY & NEBULA
        if dim_id == "GALAXY":
            if self.galaxy_bg.shape[:2] != (h, w):
                self.galaxy_bg = cv2.resize(self.galaxy_bg, (w, h))
            dim_frame = self.galaxy_bg.copy()
            # Swirling Galaxy Core
            galaxy_center = (int(self.cx), int(self.cy))
            for arm in range(3):
                arm_angle = t * 0.8 + arm * (2 * math.pi / 3.0)
                for r_step in range(15, int(self.rx * 1.1), 8):
                    ang = arm_angle + (r_step * 0.035)
                    gx = int(galaxy_center[0] + math.cos(ang) * r_step)
                    gy = int(galaxy_center[1] + math.sin(ang) * r_step)
                    if 0 <= gx < w and 0 <= gy < h:
                        cv2.circle(dim_frame, (gx, gy), random.randint(2, 5), (255, random.randint(180, 240), random.randint(0, 100)), -1)
            # Composite user silhouette as cosmic star cloud
            if mask_person is not None:
                if mask_person.shape[:2] != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_3c = cv2.cvtColor(mask_person, cv2.COLOR_GRAY2BGR) / 255.0
                dim_frame = dim_frame * (1.0 - mask_3c * 0.3) + frame * (mask_3c * 0.3)
            return np.clip(dim_frame, 0, 255).astype(np.uint8)

        # 2. 🔮 TOUCHDESIGNER PLEXUS / PROXIMITY POP
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
                                col = (203, 19, 255) if py < self.cy else (245, 248, 255)
                                nodes.append((px, py, col))

            # Fast proximity linking
            cell_size = 28
            grid = {}
            for idx, (nx, ny, col) in enumerate(nodes):
                key = (nx // cell_size, ny // cell_size)
                if key not in grid: grid[key] = []
                grid[key].append(idx)

            for idx, (x1, y1, col1) in enumerate(nodes):
                cx_c = x1 // cell_size
                cy_c = y1 // cell_size
                for gx in range(cx_c - 1, cx_c + 2):
                    for gy in range(cy_c - 1, cy_c + 2):
                        if (gx, gy) in grid:
                            for n_idx in grid[(gx, gy)]:
                                if n_idx <= idx: continue
                                x2, y2, col2 = nodes[n_idx]
                                d = math.hypot(x2 - x1, y2 - y1)
                                if d < 25.0:
                                    cv2.line(dim_frame, (x1, y1), (x2, y2), (200, 200, 255), 1)

            for x, y, col in nodes:
                cv2.circle(dim_frame, (x, y), 2, col, -1)
            return dim_frame

        # 3. ⚡ CYBERPUNK MATRIX & WIREFRAME GRID
        elif dim_id == "MATRIX":
            dim_frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Perspective 3D Neon Grid Floor
            horizon_y = h // 2
            for gy in range(horizon_y, h, 20):
                line_y = gy
                cv2.line(dim_frame, (0, line_y), (w, line_y), (0, 255, 128), 1)
            for gx in range(0, w, 35):
                cv2.line(dim_frame, (w // 2, horizon_y), (gx, h), (0, 255, 128), 1)

            # Canny edge hologram of person
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 140)
            edges_bgr = np.zeros_like(frame)
            edges_bgr[edges > 0] = (0, 255, 128)
            dim_frame = cv2.add(dim_frame, edges_bgr)
            return dim_frame

        # 4. 🌈 KALEIDOSCOPE WARP
        elif dim_id == "KALEIDO":
            quad = frame[:h//2, :w//2]
            quad_flip_h = cv2.flip(quad, 1)
            top_half = np.hstack((quad, quad_flip_h))
            bottom_half = cv2.flip(top_half, 0)
            kaleido = np.vstack((top_half, bottom_half))
            if kaleido.shape[:2] != (h, w):
                kaleido = cv2.resize(kaleido, (w, h))
            # Rainbow shift
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
            synth[:, :, 0] = np.clip(synth[:, :, 0] * 1.4, 0, 255) # Boost Blue
            synth[:, :, 2] = np.clip(synth[:, :, 2] * 1.5, 0, 255) # Boost Red
            synth[:, :, 1] = np.clip(synth[:, :, 1] * 0.4, 0, 255) # Suppress Green
            # CRT Scanlines
            for sy in range(0, h, 6):
                synth[sy, :] = synth[sy, :] // 2
            return synth

        return frame

    def draw(self, img, mask_person, hand_landmarks, face_landmarks, particles):
        h, w = img.shape[:2]
        cx, cy = int(self.cx), int(self.cy)
        rx, ry = max(10, int(self.rx)), max(10, int(self.ry))
        dim = DIMENSIONS[self.dim_idx]
        theme_col = dim["theme_color"]
        t = time.time()

        # 1. Generate Alternate Reality Dimension Frame
        dim_frame = self.render_dimension_content(img, mask_person, hand_landmarks, face_landmarks)

        # 2. Create Portal Aperture Mask
        portal_mask = np.zeros((h, w), dtype=np.uint8)
        if self.shape_idx == 0 or self.shape_idx == 1:
            # Elliptical / Circular Aperture with 3D Rotation
            cv2.ellipse(portal_mask, (cx, cy), (rx, ry), int(self.rotation_angle), 0, 360, 255, -1)
        else:
            # Hexagonal / Octagonal Polygonal Aperture
            poly_pts = []
            sides = 6
            for s in range(sides):
                ang = math.radians(self.rotation_angle + (s * 360.0 / sides))
                poly_pts.append([int(cx + math.cos(ang) * rx), int(cy + math.sin(ang) * ry)])
            cv2.fillPoly(portal_mask, [np.array(poly_pts, dtype=np.int32)], 255)

        # 3. Dimensional Composite: Replace inside of portal with alternate dimension
        mask_3c = cv2.cvtColor(portal_mask, cv2.COLOR_GRAY2BGR) / 255.0
        img[:] = (img * (1.0 - mask_3c) + dim_frame * mask_3c).astype(np.uint8)

        # 4. Dr. Strange Fiery Event Horizon Rim Sparks
        particles.emit_portal_rim_sparks(cx, cy, rx, ry, color=theme_col, count=5)

        # 5. Glowing Multi-Layer Event Horizon Rim Rings
        if self.shape_idx in [0, 1]:
            # Outer Flaming Ring
            cv2.ellipse(img, (cx, cy), (rx + 4, ry + 4), int(self.rotation_angle), 0, 360, theme_col, 3)
            # Inner White-Hot Ring
            cv2.ellipse(img, (cx, cy), (rx, ry), int(self.rotation_angle), 0, 360, (255, 255, 255), 2)
            # Rotating Rune Segments along Rim
            for r_i in range(8):
                rune_ang = self.rim_spin + r_i * (math.pi / 4.0)
                px = int(cx + math.cos(rune_ang) * (rx + 2))
                py = int(cy + math.sin(rune_ang) * (ry + 2))
                cv2.circle(img, (px, py), 4, (255, 255, 255), -1)
                cv2.circle(img, (px, py), 6, theme_col, 1)
        else:
            poly_pts = []
            sides = 6
            for s in range(sides):
                ang = math.radians(self.rotation_angle + (s * 360.0 / sides))
                poly_pts.append([int(cx + math.cos(ang) * rx), int(cy + math.sin(ang) * ry)])
            cv2.polylines(img, [np.array(poly_pts, dtype=np.int32)], True, theme_col, 4)
            cv2.polylines(img, [np.array(poly_pts, dtype=np.int32)], True, (255, 255, 255), 2)

        # 6. Portal Status & Dimension Badge
        badge_y = max(85, cy - ry - 25)
        dim_text = f"🌀 GATEWAY: {dim['name']}"
        ts = cv2.getTextSize(dim_text, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
        bx1 = max(10, cx - ts[0] // 2 - 12)
        bx2 = min(w - 10, cx + ts[0] // 2 + 12)
        overlay = img.copy()
        cv2.rectangle(overlay, (bx1, badge_y - 20), (bx2, badge_y + 8), (18, 18, 26), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (bx1, badge_y - 20), (bx2, badge_y + 8), theme_col, 2)
        cv2.putText(img, dim_text, (bx1 + 10, badge_y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)


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
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y

        tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)
        ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
        pinch_dist = math.hypot(tx - ix, ty - iy)
        if pinch_dist < 45:
            return "PINCH"

        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"

        if index_up and pinky_up and not middle_up and not ring_up:
            return "ROCK_ON"

        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM"

        if not index_up and not middle_up and not ring_up and not pinky_up:
            return "FIST"

        if index_up and not middle_up:
            return "POINTING"

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

    # Initialize Face Landmarker
    face_landmarker = None
    try:
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)
    except Exception as e:
        print("Face Landmarker status:", e)

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

    print("✨ QUANTUM SPATIAL PORTAL STUDIO ACTIVE ✨")

    last_timestamp_ms = 0
    flash_timer = 0
    gesture_cooldown = 0.0
    fps_time = time.time()
    fps = 30.0
    exit_requested = False

    particles = ParticleManager()
    portal = QuantumPortalEngine()

    while not exit_requested:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        if portal.w != w or portal.h != h:
            portal.w, portal.h = w, h

        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        face_results = None
        if face_landmarker is not None:
            face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        mask_person = None
        if segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

        # ----------------- 1. UPDATE PORTAL SPATIAL GEOMETRY -----------------
        portal.update_geometry(hand_results.hand_landmarks if hand_results else None, w, h, particles)

        # ----------------- 2. RENDER THE QUANTUM PORTAL & ALTERNATE DIMENSION -----------------
        portal.draw(
            img,
            mask_person,
            hand_results.hand_landmarks if hand_results else None,
            face_results.face_landmarks if face_results else None,
            particles
        )

        # ----------------- 3. TOUCHLESS SPATIAL BUTTONS -----------------
        buttons = []
        buttons.append(SpatialButton("DIM_0", 10, 10, 80, 38, "🌌 GALAXY", border_color=(255, 200, 0)))
        buttons.append(SpatialButton("DIM_1", 94, 10, 82, 38, "🔮 PLEXUS", border_color=(203, 19, 255)))
        buttons.append(SpatialButton("DIM_2", 180, 10, 82, 38, "⚡ MATRIX", border_color=(0, 255, 128)))
        buttons.append(SpatialButton("DIM_3", 266, 10, 84, 38, "🌈 KALEIDO", border_color=(255, 0, 200)))
        buttons.append(SpatialButton("DIM_4", 354, 10, 84, 38, "🧊 THERMAL", border_color=(0, 140, 255)))
        buttons.append(SpatialButton("DIM_5", 442, 10, 84, 38, "👾 80s RETRO", border_color=(255, 60, 180)))

        # Shape Toggle
        shape_lbl = f"🔲 {PORTAL_SHAPES[portal.shape_idx].split()[0]}"
        buttons.append(SpatialButton("TOGGLE_SHAPE", 532, 10, 88, 38, shape_lbl, border_color=(0, 255, 255)))

        # Universal Action Buttons
        buttons.append(SpatialButton("PHOTO", w - 158, 10, 76, 38, "📸 SNAP", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("EXIT", w - 80, 10, 72, 38, "❌ EXIT", border_color=(0, 0, 255), hold_time=0.9))

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

                # Peace Sign = Instant Dimension Jump
                if g_type == "PEACE" and time.time() > gesture_cooldown:
                    portal.next_dimension(particles)
                    gesture_cooldown = time.time() + 0.9

            # Two Hands Touch Fingertips = Next Dimension
            if len(pointers) >= 2:
                if math.hypot(pointers[0][0] - pointers[1][0], pointers[0][1] - pointers[1][1]) < 35 and time.time() > gesture_cooldown:
                    portal.next_dimension(particles)
                    gesture_cooldown = time.time() + 0.9

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
                if btn.bid.startswith("DIM_") and btn.bid[4:].isdigit():
                    portal.set_dimension_by_index(int(btn.bid[4:]), particles)
                elif btn.bid == "TOGGLE_SHAPE":
                    portal.next_shape(particles)
                elif btn.bid == "PHOTO":
                    fname = f"quantum_portal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fpath = os.path.join(SCREENSHOTS_DIR, fname)
                    cv2.imwrite(fpath, img)
                    print(f"📸 Photo saved: {fpath}")
                    flash_timer = 4
                elif btn.bid == "EXIT":
                    exit_requested = True

        # Draw Futuristic Interactive Cursors at Index Fingertips
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
        cv2.line(img, (0, 56), (w, 56), DIMENSIONS[portal.dim_idx]["theme_color"], 2)

        for btn in buttons:
            is_active = False
            if btn.bid.startswith("DIM_") and btn.bid[4:].isdigit():
                is_active = int(btn.bid[4:]) == portal.dim_idx
            btn.draw(img, is_active=is_active)

        # Bottom Bar & Telemetry
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        status_text = "🌀 QUANTUM SPATIAL PORTAL: 🖐️ Move hands apart to expand portal | ✌️ Peace Sign = Switch Universe | 👉 Hover/Pinch buttons!"
        cv2.putText(img, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

        now = time.time()
        fps = 1.0 / max(0.001, now - fps_time)
        fps_time = now
        cv2.putText(img, f"FPS: {int(fps)}", (w - 85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if flash_timer > 0:
            cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
            flash_timer -= 1

        cv2.imshow('Quantum Spatial Portal & Dimensional Gateway Studio', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()