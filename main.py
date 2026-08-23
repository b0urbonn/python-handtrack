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
DALE_PHOTO_PATH = os.path.join(SCRIPT_DIR, 'dale_photo.png')
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, 'screenshots')
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ----------------- CONSTANTS & CONFIG -----------------
FILTERS = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"]
MODES = ["NARUTO JUTSU", "3D MATTER", "TD CLOAK", "IRON MAN", "BALL GAME", "AIR CANVAS", "RETRO PORTAL"]

# MediaPipe Hand Skeleton Connection Map (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm cross-connections
]

FACE_FX_MODES = ["SHARINGAN", "MANGEKYO", "FACE DOTS", "CYBER MESH", "CYBER VISOR", "IRON MAN HUD", "NEON CROWN", "CAT EARS", "OFF"]
BALL_TYPES = ["NEON", "FIREBALL", "PLASMA"]
MATTER_STATES = ["SOLID", "LIQUID", "GAS", "PLASMA"]
MATTER_SHAPES = ["SPHERE", "TORUS", "CUBE", "HEART", "HELIX"]
TD_CLOAK_MODES = ["PREDATOR CAMO", "POINT CLOUD DISSOLVE", "HOLLOW MATRIX", "QUANTUM GHOST"]
NARUTO_JUTSUS = ["AUTO GESTURE", "CHIDORI ⚡", "RASENGAN 🌀", "KATON FIRE 🔥", "RASENSHURIKEN 🟣", "KIRIN DRAGON 🐉"]

PALETTE = [
    {"name": "CYAN", "color": (255, 230, 0), "type": "solid"},
    {"name": "PINK", "color": (203, 19, 255), "type": "solid"},
    {"name": "GREEN", "color": (50, 255, 50), "type": "solid"},
    {"name": "YELLOW", "color": (0, 230, 255), "type": "solid"},
    {"name": "ORANGE", "color": (0, 140, 255), "type": "solid"},
    {"name": "RED", "color": (50, 50, 255), "type": "solid"},
    {"name": "RAINBOW", "color": "RAINBOW", "type": "rainbow"},
    {"name": "FIRE", "color": (0, 165, 255), "type": "fire"},
    {"name": "ERASER", "color": (0, 0, 0), "type": "eraser"},
]

LIP_INDICES = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
EYE_INDICES = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 468, 473}
EYEBROW_INDICES = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285}
NOSE_INDICES = {1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275, 48, 278, 219, 439, 218, 438}

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]


# ----------------- VISUAL SHOCKWAVE -----------------
class Shockwave:
    def __init__(self, x, y, color=(0, 255, 255), max_radius=60):
        self.x = int(x)
        self.y = int(y)
        self.color = color
        self.radius = 6.0
        self.max_radius = float(max_radius)
        self.life = 1.0

    def update(self):
        self.radius += 5.0
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
        elif self.p_type == "fire":
            self.vy -= 0.45
            self.vx += random.uniform(-0.6, 0.6)
        elif self.p_type == "lightning":
            self.vx += random.uniform(-1.2, 1.2)
            self.vy += random.uniform(-1.2, 1.2)
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
        if len(self.particles) < 750:
            self.particles.append(p)

    def add_shockwave(self, x, y, color=(0, 255, 255), max_radius=60):
        self.shockwaves.append(Shockwave(x, y, color, max_radius))

    def emit_sparkles(self, x, y, color=(0, 255, 255), count=3, speed=3):
        for _ in range(count):
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(1, speed)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, color, random.uniform(2, 5), random.randint(15, 30)))

    def emit_lightning_sparks(self, x, y, count=4):
        colors = [(255, 255, 255), (255, 220, 0), (255, 180, 0), (200, 255, 255)]
        for _ in range(count):
            c = random.choice(colors)
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(3, 9)
            self.add_particle(Particle(x, y, math.cos(angle) * spd, math.sin(angle) * spd, c, random.uniform(2, 5), random.randint(8, 18), decay=0.92, p_type="lightning"))

    def emit_fire_particles(self, x, y, count=4):
        fire_colors = [(0, 69, 255), (0, 140, 255), (0, 215, 255), (0, 255, 255)]
        for _ in range(count):
            c = random.choice(fire_colors)
            self.add_particle(Particle(
                x + random.uniform(-6, 6), y + random.uniform(-6, 6),
                random.uniform(-1.2, 1.2), random.uniform(-3.5, -1.0),
                c, random.uniform(4, 9), random.randint(14, 28), decay=0.94, p_type="fire"
            ))

    def emit_fireworks(self, x, y, count=45):
        colors = [(0, 165, 255), (255, 255, 0), (255, 50, 50), (50, 255, 50), (255, 0, 255), (0, 255, 255)]
        for _ in range(count):
            c = random.choice(colors)
            angle = random.uniform(0, math.pi * 2)
            spd = random.uniform(3, 11)
            vx = math.cos(angle) * spd
            vy = math.sin(angle) * spd
            self.add_particle(Particle(x, y, vx, vy, c, random.uniform(3, 7), random.randint(25, 50), decay=0.98, p_type="firework"))
        self.add_shockwave(x, y, (255, 255, 255), max_radius=90)

    def emit_laser_beam(self, x1, y1, x2, y2, color=(255, 0, 200)):
        for _ in range(16):
            t = random.random()
            px = x1 + (x2 - x1) * t + random.uniform(-4, 4)
            py = y1 + (y2 - y1) * t + random.uniform(-4, 4)
            self.add_particle(Particle(px, py, random.uniform(-1, 1), random.uniform(-1, 1), color, random.uniform(2, 4), random.randint(8, 18)))

    def update_and_draw(self, img):
        self.particles = [p for p in self.particles if p.update() and (p.draw(img) or True)]
        self.shockwaves = [sw for sw in self.shockwaves if sw.update() and (sw.draw(img) or True)]


# ----------------- NARUTO ELEMENTAL JUTSU ENGINE -----------------
class NarutoJutsuEngine:
    """Simulates iconic Naruto Jutsu: Chidori (Lightning), Rasengan (Spiral Sphere), Katon (Fireball), and Rasenshuriken."""
    def __init__(self):
        self.active_jutsu = "AUTO GESTURE"
        self.chakra_time = 0.0
        self.lightning_branches = []

    def draw_lightning_bolt(self, img, p1, p2, color=(255, 240, 0), thickness=2, subdivisions=4, max_displacement=16):
        """Draws a jagged procedural electric lightning bolt between two points."""
        pts = [p1, p2]
        for _ in range(subdivisions):
            new_pts = []
            for i in range(len(pts) - 1):
                a, b = pts[i], pts[i + 1]
                mid_x = (a[0] + b[0]) // 2 + random.randint(-max_displacement, max_displacement)
                mid_y = (a[1] + b[1]) // 2 + random.randint(-max_displacement, max_displacement)
                new_pts.extend([a, (mid_x, mid_y)])
            new_pts.append(pts[-1])
            pts = new_pts
            max_displacement = max(4, max_displacement // 2)

        for i in range(len(pts) - 1):
            # Outer cyan/blue lightning glow
            cv2.line(img, pts[i], pts[i + 1], (255, 200, 0), thickness + 3)
            # Inner white lightning core
            cv2.line(img, pts[i], pts[i + 1], (255, 255, 255), max(1, thickness))

    def render_chidori(self, img, palm_pt, fingertips, particles):
        """⚡ CHIDORI (Lightning Blade / Raikiri): Roaring high-voltage electric lightning arcs."""
        px, py = palm_pt
        t = time.time()
        
        # Roaring central electric sphere
        core_r = int(24 + 6 * math.sin(t * 20))
        cv2.circle(img, (px, py), core_r + 15, (255, 200, 0), 2)
        cv2.circle(img, (px, py), core_r, (255, 230, 100), -1)
        cv2.circle(img, (px, py), core_r // 2, (255, 255, 255), -1)

        # Arcs shooting to fingertips
        for ft in fingertips:
            self.draw_lightning_bolt(img, (px, py), ft, color=(255, 240, 0), thickness=2, subdivisions=3, max_displacement=14)
            if random.random() < 0.4:
                particles.emit_lightning_sparks(ft[0], ft[1], count=2)

        # Wild branching lightning bolts shooting out from hand
        for _ in range(6):
            angle = random.uniform(0, math.pi * 2)
            length = random.uniform(60, 150)
            end_x = int(px + math.cos(angle) * length)
            end_y = int(py + math.sin(angle) * length)
            self.draw_lightning_bolt(img, (px, py), (end_x, end_y), color=(255, 240, 0), thickness=2, subdivisions=3, max_displacement=18)

        particles.emit_lightning_sparks(px, py, count=4)
        cv2.putText(img, "⚡ CHIDORI (LIGHTNING BLADE)", (px - 90, py - core_r - 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 240, 0), 2)

    def render_rasengan(self, img, palm_pt, particles):
        """🌀 RASENGAN (Spiraling Sphere): Concentric high-speed swirling chakra vortex."""
        px, py = palm_pt
        self.chakra_time += 0.25
        t = self.chakra_time
        base_r = 46

        # Swirling Concentric Chakra Rings
        for ring_i in range(5):
            r_val = base_r - ring_i * 7
            angle_rot = t * (3.5 + ring_i * 1.2)
            axes = (r_val, int(r_val * 0.65))
            cv2.ellipse(img, (px, py), axes, int(math.degrees(angle_rot)), 0, 360, (255, 230, 0), 2)
            cv2.ellipse(img, (px, py), (axes[1], axes[0]), int(math.degrees(-angle_rot)), 0, 360, (255, 180, 0), 2)

        # Dense Glowing Cyan Core
        cv2.circle(img, (px, py), 22, (255, 240, 150), -1)
        cv2.circle(img, (px, py), 12, (255, 255, 255), -1)

        # Outer Chakra Wind Aura
        for _ in range(4):
            sp_angle = random.uniform(0, math.pi * 2)
            sp_r = random.uniform(base_r, base_r + 25)
            sx = int(px + math.cos(sp_angle) * sp_r)
            sy = int(py + math.sin(sp_angle) * sp_r)
            cv2.circle(img, (sx, sy), random.randint(2, 4), (255, 255, 0), -1)

        if random.random() < 0.4:
            particles.emit_sparkles(px, py, (255, 240, 0), count=2, speed=3)

        cv2.putText(img, "🌀 RASENGAN (CHAKRA VORTEX)", (px - 90, py - base_r - 28), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 240, 0), 2)

    def render_katon_fireball(self, img, seal_pt, particles):
        """🔥 KATON: FIRE STYLE FIREBALL (Gōkakyū no Jutsu): Roaring stream of flaming inferno."""
        sx, sy = seal_pt
        t = time.time()
        
        # Fireball stream trajectory (bursting upward & outward)
        fire_r = int(55 + 12 * math.sin(t * 15))
        fx = sx
        fy = max(80, sy - 80)

        # Layered Solar Inferno Flame
        cv2.circle(img, (fx, fy), fire_r + 20, (0, 69, 255), -1)     # Red-Orange outer
        cv2.circle(img, (fx, fy), fire_r + 5, (0, 140, 255), -1)     # Orange mid
        cv2.circle(img, (fx, fy), fire_r - 12, (0, 230, 255), -1)    # Yellow inner
        cv2.circle(img, (fx, fy), fire_r // 3, (255, 255, 255), -1)  # White hot core

        # Fire stream connecting hand to fireball
        cv2.line(img, (sx, sy), (fx, fy), (0, 165, 255), 14)
        cv2.line(img, (sx, sy), (fx, fy), (0, 230, 255), 6)
        cv2.line(img, (sx, sy), (fx, fy), (255, 255, 255), 2)

        particles.emit_fire_particles(fx, fy, count=6)
        particles.emit_fire_particles(sx, sy, count=2)

        cv2.putText(img, "🔥 KATON: FIREBALL JUTSU", (fx - 90, fy - fire_r - 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 165, 255), 2)

    def render_rasenshuriken(self, img, center_pt, particles):
        """🟣 WIND STYLE: RASENSHURIKEN: Giant rotating 4-blade sonic chakra shuriken."""
        cx, cy = center_pt
        self.chakra_time += 0.35
        t = self.chakra_time
        shuriken_r = 95

        # 4 High-speed rotating curved wind chakra blades
        for blade_i in range(4):
            b_angle = t * 4.0 + blade_i * (math.pi / 2.0)
            tip_x = int(cx + math.cos(b_angle) * shuriken_r)
            tip_y = int(cy + math.sin(b_angle) * shuriken_r)

            wing1_x = int(cx + math.cos(b_angle + 0.35) * (shuriken_r * 0.45))
            wing1_y = int(cy + math.sin(b_angle + 0.35) * (shuriken_r * 0.45))
            wing2_x = int(cx + math.cos(b_angle - 0.35) * (shuriken_r * 0.45))
            wing2_y = int(cy + math.sin(b_angle - 0.35) * (shuriken_r * 0.45))

            blade_poly = np.array([[cx, cy], [wing1_x, wing1_y], [tip_x, tip_y], [wing2_x, wing2_y]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [blade_poly], (255, 0, 200))
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.polylines(img, [blade_poly], True, (255, 255, 255), 2)

        # Central Rasengan sphere
        cv2.circle(img, (cx, cy), 32, (255, 200, 0), -1)
        cv2.circle(img, (cx, cy), 16, (255, 255, 255), -1)
        particles.emit_sparkles(cx, cy, (255, 0, 200), count=4, speed=5)

        cv2.putText(img, "🟣 RASENSHURIKEN (WIND STYLE)", (cx - 105, cy - shuriken_r - 20), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 0, 200), 2)


# ----------------- TOUCHDESIGNER INVISIBILITY & GENERATIVE FX ENGINE -----------------
class TouchDesignerFX:
    def __init__(self):
        self.bg_frame = None
        self.mode_idx = 0
        self.flow_time = 0.0

    def update_background(self, frame, mask=None):
        if self.bg_frame is None:
            self.bg_frame = frame.astype(np.float32)
        elif mask is not None:
            bg_mask = (mask == 0)
            if np.any(bg_mask):
                self.bg_frame[bg_mask] = self.bg_frame[bg_mask] * 0.95 + frame[bg_mask].astype(np.float32) * 0.05
        else:
            self.bg_frame = self.bg_frame * 0.98 + frame.astype(np.float32) * 0.02

    def next_mode(self, particles=None):
        self.mode_idx = (self.mode_idx + 1) % len(TD_CLOAK_MODES)
        if particles:
            particles.add_shockwave(640, 360, (0, 255, 255), max_radius=90)

    def render(self, frame, mask_person, particles):
        if mask_person is None or self.bg_frame is None:
            return frame

        h, w = frame.shape[:2]
        bg = self.bg_frame.astype(np.uint8)
        if bg.shape[:2] != (h, w):
            bg = cv2.resize(bg, (w, h))

        cur_mode = TD_CLOAK_MODES[self.mode_idx]
        t = time.time()
        self.flow_time += 0.06

        if cur_mode == "PREDATOR CAMO":
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(mask_person, kernel, iterations=2)
            edge_mask = cv2.subtract(dilated, mask_person)

            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)
            for y in range(0, h, 8):
                sin_y = math.sin(y * 0.05 + t * 4.0) * 12.0
                map_y[y:y+8, :] = np.arange(y, min(h, y+8))[:, None]
                map_x[y:y+8, :] = np.arange(w)[None, :] + sin_y

            warped_bg = cv2.remap(bg, map_x, map_y, cv2.INTER_LINEAR)
            mask_3c = cv2.cvtColor(mask_person, cv2.COLOR_GRAY2BGR) / 255.0
            edge_3c = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR) / 255.0
            
            result = frame * (1.0 - mask_3c) + bg * mask_3c
            result = result * (1.0 - edge_3c) + warped_bg * edge_3c
            result = np.clip(result, 0, 255).astype(np.uint8)

            edge_glow = np.zeros_like(frame)
            edge_glow[edge_mask > 0] = (0, 255, 255)
            edge_glow = cv2.GaussianBlur(edge_glow, (15, 15), 0)
            return cv2.addWeighted(result, 1.0, edge_glow, 0.45, 0)

        elif cur_mode == "POINT CLOUD DISSOLVE":
            result = bg.copy()
            y_indices, x_indices = np.where(mask_person > 0)
            if len(x_indices) > 0:
                sample_count = min(750, len(x_indices))
                chosen = np.random.choice(len(x_indices), sample_count, replace=False)
                for c_idx in chosen:
                    px = x_indices[c_idx]
                    py = y_indices[c_idx]
                    noise_x = math.sin(py * 0.03 + self.flow_time) * 18.0
                    noise_y = math.cos(px * 0.03 + self.flow_time) * 18.0 - 8.0
                    fx = int(px + noise_x)
                    fy = int(py + noise_y)
                    if 0 <= fx < w and 0 <= fy < h:
                        color_sample = frame[py, px]
                        glow_col = (int(color_sample[0] * 0.3 + 200 * 0.7),
                                    int(color_sample[1] * 0.3 + 240 * 0.7),
                                    int(color_sample[2] * 0.3 + 0 * 0.7))
                        cv2.circle(result, (fx, fy), 2, glow_col, -1)
            return result

        elif cur_mode == "HOLLOW MATRIX":
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.Canny(mask_person, 100, 200)
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)

            mask_3c = cv2.cvtColor(mask_person, cv2.COLOR_GRAY2BGR) / 255.0
            result = frame * (1.0 - mask_3c) + bg * mask_3c

            overlay = result.copy()
            scan_y_offset = int((t * 50) % 12)
            for sy in range(scan_y_offset, h, 12):
                scan_mask = (mask_person[sy, :] > 0)
                overlay[sy, scan_mask] = (0, 255, 255)
            result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)
            result[edges_dilated > 0] = (255, 0, 200)
            return result

        elif cur_mode == "QUANTUM GHOST":
            ghost_body = frame.copy()
            ghost_cyan = ghost_body.copy()
            ghost_cyan[:, :, 2] = 0
            ghost_magenta = ghost_body.copy()
            ghost_magenta[:, :, 1] = 0

            shift = int(8 + 4 * math.sin(t * 5.0))
            ghost_composite = np.zeros_like(frame)
            ghost_composite[:, :-shift] = ghost_cyan[:, shift:]
            ghost_composite[:, shift:] = cv2.add(ghost_composite[:, shift:], ghost_magenta[:, :-shift])

            mask_3c = cv2.cvtColor(mask_person, cv2.COLOR_GRAY2BGR) / 255.0
            result = bg * (1.0 - mask_3c * 0.4) + ghost_composite * (mask_3c * 0.6)
            return np.clip(result, 0, 255).astype(np.uint8)

        return frame


# ----------------- REAL 3D VOLUMETRIC SPHERE & QUANTUM MATTER SIMULATOR -----------------
class QuantumMatterSimulator:
    NUM_PARTICLES = 360

    def __init__(self, center_x=640, center_y=360):
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.target_x = float(center_x)
        self.target_y = float(center_y)
        self.radius = 145.0
        self.scale = 1.0
        self.state_idx = 0
        self.shape_idx = 0
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.spin_x = 0.015
        self.spin_y = 0.022
        self.liquid_wave = 0.0
        self.locked = False
        self.last_lock_toggle = 0.0
        self.morph_progress = 1.0
        self.prev_shape_idx = 0
        self.squish_active = False
        self.squish_axis = np.array([1.0, 0.0], dtype=np.float32)
        self.squish_amount = 1.0
        self.hand2_pt = None

        self.base_shapes = {
            "SPHERE": self._generate_sphere(self.NUM_PARTICLES),
            "TORUS": self._generate_torus(self.NUM_PARTICLES),
            "CUBE": self._generate_cube(self.NUM_PARTICLES),
            "HEART": self._generate_heart(self.NUM_PARTICLES),
            "HELIX": self._generate_helix(self.NUM_PARTICLES)
        }

        self.particles_pos = np.copy(self.base_shapes["SPHERE"])
        self.particles_vel = np.zeros((self.NUM_PARTICLES, 3), dtype=np.float32)
        self.particle_phases = np.random.uniform(0, math.pi * 2, self.NUM_PARTICLES)

    def _generate_sphere(self, n):
        pts = np.zeros((n, 3), dtype=np.float32)
        phi = math.pi * (math.sqrt(5.0) - 1.0)
        for i in range(n):
            layer = 1.0 if i < n * 0.75 else 0.55
            y = (1.0 - (i / float(n - 1)) * 2.0) * layer
            radius = math.sqrt(max(0.0, layer * layer - y * y))
            theta = phi * i
            pts[i] = [math.cos(theta) * radius, y, math.sin(theta) * radius]
        return pts

    def _generate_torus(self, n, R=0.82, r=0.36):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            u = (i / float(n)) * math.pi * 2 * 6.0
            v = (i / float(n)) * math.pi * 2
            x = (R + r * math.cos(u)) * math.cos(v)
            y = (R + r * math.cos(u)) * math.sin(v)
            z = r * math.sin(u)
            pts[i] = [x, y, z]
        return pts

    def _generate_cube(self, n):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            face = random.randint(0, 5)
            u = random.uniform(-0.8, 0.8)
            v = random.uniform(-0.8, 0.8)
            if face == 0: pts[i] = [0.8, u, v]
            elif face == 1: pts[i] = [-0.8, u, v]
            elif face == 2: pts[i] = [u, 0.8, v]
            elif face == 3: pts[i] = [u, -0.8, v]
            elif face == 4: pts[i] = [u, v, 0.8]
            else: pts[i] = [u, v, -0.8]
        return pts

    def _generate_heart(self, n):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            t = (i / float(n)) * math.pi * 2
            x = 16 * (math.sin(t) ** 3) / 16.0
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 16.0
            z = math.sin(t * 3.0) * 0.35
            pts[i] = [x * 0.95, y * 0.95, z]
        return pts

    def _generate_helix(self, n):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            strand = 1 if (i % 2 == 0) else -1
            t = (i / float(n)) * math.pi * 6 - math.pi * 3
            if i % 8 == 0:
                rung_t = random.uniform(-0.6, 0.6)
                pts[i] = [math.cos(t) * rung_t, t / 3.8, math.sin(t) * rung_t]
            else:
                pts[i] = [math.cos(t) * 0.65 * strand, t / 3.8, math.sin(t) * 0.65 * strand]
        return pts

    def toggle_lock(self, particles=None):
        now = time.time()
        if now - self.last_lock_toggle > 0.6:
            self.locked = not self.locked
            self.last_lock_toggle = now
            if particles:
                col = (0, 255, 0) if self.locked else (0, 255, 255)
                particles.add_shockwave(int(self.center_x), int(self.center_y), col, max_radius=85)

    def set_state(self, state_name, particles=None):
        if state_name in MATTER_STATES:
            self.state_idx = MATTER_STATES.index(state_name)
            if particles:
                col = (0, 255, 255) if state_name == "SOLID" else (255, 180, 0) if state_name == "LIQUID" else (255, 0, 255)
                particles.add_shockwave(int(self.center_x), int(self.center_y), col, max_radius=85)

    def set_shape_by_index(self, s_idx, particles=None):
        if 0 <= s_idx < len(MATTER_SHAPES) and s_idx != self.shape_idx:
            self.prev_shape_idx = self.shape_idx
            self.shape_idx = s_idx
            self.morph_progress = 0.0
            if particles:
                particles.emit_fireworks(int(self.center_x), int(self.center_y), count=30)

    def next_shape(self, particles=None):
        next_i = (self.shape_idx + 1) % len(MATTER_SHAPES)
        self.set_shape_by_index(next_i, particles)

    def update(self, hand_landmarks, w, h, particles):
        current_state = MATTER_STATES[self.state_idx]
        current_shape = MATTER_SHAPES[self.shape_idx]
        base_target = self.base_shapes[current_shape]

        if self.morph_progress < 1.0:
            self.morph_progress = min(1.0, self.morph_progress + 0.08)

        if hand_landmarks and len(hand_landmarks) > 0:
            lms = hand_landmarks[0]
            hx, hy = int(lms[9].x * w), int(lms[9].y * h)
            
            if not self.locked:
                self.target_x = hx
                self.target_y = hy

            dx_hand = lms[9].x - lms[0].x
            dy_hand = lms[9].y - lms[0].y
            self.spin_y = dx_hand * 0.08
            self.spin_x = -dy_hand * 0.08

            tx, ty = int(lms[4].x * w), int(lms[4].y * h)
            ix, iy = int(lms[8].x * w), int(lms[8].y * h)
            pinch_gap = math.hypot(tx - ix, ty - iy)
            self.scale = max(0.55, min(1.8, pinch_gap / 75.0))

            mx, my = int(lms[12].x * w), int(lms[12].y * h)
            two_finger_dist = math.hypot(ix - mx, iy - my)
            if (pinch_gap < 32 or two_finger_dist < 28) and time.time() - self.last_lock_toggle > 0.8:
                self.toggle_lock(particles)

            finger_count, index_up, middle_up, ring_up, pinky_up, thumb_up = GestureRecognizer.count_fingers(lms)
            if finger_count == 1:
                self.set_shape_by_index(0, particles)
            elif finger_count == 2:
                self.set_shape_by_index(1, particles)
            elif finger_count == 3:
                self.set_shape_by_index(2, particles)
            elif finger_count == 4:
                self.set_shape_by_index(3, particles)
            elif finger_count == 5:
                self.set_shape_by_index(4, particles)
        else:
            self.spin_x = 0.012
            self.spin_y = 0.018

        self.squish_active = False
        self.squish_amount = 1.0
        self.hand2_pt = None

        if hand_landmarks and len(hand_landmarks) >= 2:
            lms2 = hand_landmarks[1]
            h2_x, h2_y = int(lms2[9].x * w), int(lms2[9].y * h)
            self.hand2_pt = (h2_x, h2_y)

            dx2 = h2_x - self.center_x
            dy2 = h2_y - self.center_y
            dist2 = math.hypot(dx2, dy2)
            max_interact_dist = self.radius * self.scale * 1.6

            if dist2 < max_interact_dist and dist2 > 1.0:
                self.squish_active = True
                self.squish_axis = np.array([dx2 / dist2, dy2 / dist2], dtype=np.float32)
                self.squish_amount = max(0.35, min(1.0, dist2 / max_interact_dist))

                if current_state in ["LIQUID", "GAS"] and random.random() < 0.25:
                    particles.emit_sparkles(h2_x, h2_y, (255, 200, 0), count=2, speed=3)

        self.center_x += (self.target_x - self.center_x) * 0.22
        self.center_y += (self.target_y - self.center_y) * 0.22

        self.rot_x += self.spin_x
        self.rot_y += self.spin_y
        self.rot_z += 0.005

        t = time.time()
        self.liquid_wave += 0.08

        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        cz, sz = math.cos(self.rot_z), math.sin(self.rot_z)

        for i in range(self.NUM_PARTICLES):
            bx, by, bz = base_target[i]

            if self.morph_progress < 1.0:
                vortex_angle = (1.0 - self.morph_progress) * math.pi * 4.0 + i
                vortex_r = (1.0 - self.morph_progress) * 1.5
                bx += math.cos(vortex_angle) * vortex_r
                by += math.sin(vortex_angle) * vortex_r

            rx1 = bx * cy + bz * sy
            ry1 = by
            rz1 = -bx * sy + bz * cy

            if current_state == "LIQUID":
                wave = math.sin(self.liquid_wave + self.particle_phases[i]) * 0.18
                rx1 *= (1.0 + wave)
                ry1 += math.sin(t * 3.0 + bx * 4.0) * 0.15 + 0.10

            rx2 = rx1
            ry2 = ry1 * cx - rz1 * sx
            rz2 = ry1 * sx + rz1 * cx

            rx3 = rx2 * cz - ry2 * sz
            ry3 = rx2 * sz + ry2 * cz
            rz3 = rz2

            target_pt = np.array([rx3, ry3, rz3], dtype=np.float32)

            if current_state == "SOLID":
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.35
            elif current_state == "LIQUID":
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.18
                self.particles_pos[i] += np.random.normal(0, 0.015, 3)
            elif current_state == "GAS":
                gas_expand = 1.6 + math.sin(t * 2.0 + self.particle_phases[i]) * 0.4
                g_target = target_pt * gas_expand
                g_target[1] -= 0.35
                self.particles_pos[i] += (g_target - self.particles_pos[i]) * 0.08
                self.particles_pos[i] += np.random.normal(0, 0.035, 3)
            elif current_state == "PLASMA":
                plasma_spin = t * 6.0 + self.particle_phases[i]
                p_target = np.array([math.cos(plasma_spin) * 0.9, math.sin(plasma_spin) * 0.9, rz3 * 1.3], dtype=np.float32)
                self.particles_pos[i] += (p_target - self.particles_pos[i]) * 0.25

    def draw(self, img, particles):
        current_state = MATTER_STATES[self.state_idx]
        current_shape = MATTER_SHAPES[self.shape_idx]
        cx, cy = int(self.center_x), int(self.center_y)
        r_current = self.radius * self.scale

        projected = []
        for i in range(self.NUM_PARTICLES):
            px, py, pz = self.particles_pos[i]
            
            if self.squish_active:
                ax, ay = self.squish_axis[0], self.squish_axis[1]
                dot = px * ax + py * ay
                perp_x, perp_y = -ay, ax
                dot_perp = px * perp_x + py * perp_y
                px = (dot * self.squish_amount) * ax + (dot_perp * (1.0 / math.sqrt(self.squish_amount))) * perp_x
                py = (dot * self.squish_amount) * ay + (dot_perp * (1.0 / math.sqrt(self.squish_amount))) * perp_y

            camera_dist = 3.5
            factor = camera_dist / (camera_dist + pz)
            sx = int(cx + px * r_current * factor)
            sy = int(cy + py * r_current * factor)
            projected.append((sx, sy, pz, factor, px, py, i))

        projected.sort(key=lambda p: p[2])

        if current_state == "SOLID":
            base_bgr = (255, 230, 0)
        elif current_state == "LIQUID":
            base_bgr = (255, 170, 0)
        elif current_state == "GAS":
            base_bgr = (203, 19, 255)
        else:
            base_bgr = (0, 140, 255)

        if current_state == "SOLID":
            for i in range(0, min(90, len(projected))):
                p1 = projected[i]
                for j in range(i + 1, min(i + 5, len(projected))):
                    p2 = projected[j]
                    dist_2d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist_2d < 45 * self.scale:
                        alpha = max(0.1, 1.0 - (dist_2d / (45 * self.scale)))
                        line_col = (int(base_bgr[0] * alpha), int(base_bgr[1] * alpha), int(base_bgr[2] * alpha))
                        cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), line_col, 1)

        for sx, sy, pz, factor, px, py, idx in projected:
            depth_alpha = max(0.25, min(1.0, (pz + 1.6) / 3.2))
            sphere_r = max(3, int(6 * factor * self.scale))

            lx, ly, lz = 0.577, -0.577, 0.577
            nx, ny, nz = px, py, pz
            norm = max(0.001, math.hypot(nx, math.hypot(ny, nz)))
            nx, ny, nz = nx / norm, ny / norm, nz / norm
            diffuse = max(0.2, (nx * lx + ny * ly + nz * lz))
            
            c_b = int(min(255, base_bgr[0] * diffuse * depth_alpha))
            c_g = int(min(255, base_bgr[1] * diffuse * depth_alpha))
            c_r = int(min(255, base_bgr[2] * diffuse * depth_alpha))
            
            cv2.circle(img, (sx, sy), sphere_r, (c_b, c_g, c_r), -1)
            spec_r = max(1, sphere_r // 3)
            cv2.circle(img, (sx - sphere_r // 3, sy - sphere_r // 3), spec_r, (255, 255, 255), -1)

        if self.squish_active and self.hand2_pt:
            h2x, h2y = self.hand2_pt
            cv2.line(img, (cx, cy), (h2x, h2y), (0, 255, 255), 2)
            cv2.circle(img, (h2x, h2y), 18, (0, 200, 255), 2)
            cv2.putText(img, "💥 SQUISHING", (h2x - 35, h2y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        hud_x, hud_y = cx - 130, cy - int(r_current) - 45
        overlay = img.copy()
        cv2.rectangle(overlay, (hud_x - 10, hud_y - 22), (hud_x + 260, hud_y + 35), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)
        border_col = (0, 255, 0) if self.locked else (0, 255, 255)
        cv2.rectangle(img, (hud_x - 10, hud_y - 22), (hud_x + 260, hud_y + 35), border_col, 1)

        lock_badge = "🔒 LOCKED" if self.locked else "🔓 TRACKING"
        state_icon = "🧊" if current_state == "SOLID" else "💧" if current_state == "LIQUID" else "💨" if current_state == "GAS" else "⚡"
        status_line = f"{state_icon} {current_state} | {current_shape} | {lock_badge}"
        cv2.putText(img, status_line, (hud_x, hud_y), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(img, "☝️1=Sphere ✌️2=Torus 🤟3=Cube 🖐️4=Heart 🧬5=Helix", (hud_x, hud_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.putText(img, "👌 2-Finger Pinch = Lock | 🖐️ 2nd Hand = Squish", (hud_x, hud_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1)


# ----------------- FULL 478-POINT FACE MESH & OCULAR JUTSU -----------------
class FaceFXRenderer:
    @staticmethod
    def render_face_fx(img, face_landmarks, fx_mode, particles):
        if not face_landmarks or fx_mode == "OFF":
            return

        h, w = img.shape[:2]
        lms = face_landmarks[0]
        p_left_pupil = (int(lms[468].x * w), int(lms[468].y * h)) if len(lms) > 468 else (int(lms[33].x * w), int(lms[33].y * h))
        p_right_pupil = (int(lms[473].x * w), int(lms[473].y * h)) if len(lms) > 473 else (int(lms[263].x * w), int(lms[263].y * h))
        t = time.time()

        # 1. 👁️ SHARINGAN (3-Tomoe Spinning Blood Red Iris)
        if fx_mode in ["SHARINGAN", "MANGEKYO"]:
            for pupil in [p_left_pupil, p_right_pupil]:
                # Glowing Blood-Red Iris
                cv2.circle(img, pupil, 16, (0, 0, 255), -1)
                cv2.circle(img, pupil, 16, (0, 0, 180), 2)
                cv2.circle(img, pupil, 4, (0, 0, 0), -1)

                if fx_mode == "SHARINGAN":
                    # 3 Tomoe spinning
                    for t_i in range(3):
                        tomoe_angle = t * 6.0 + t_i * (2 * math.pi / 3.0)
                        tx = int(pupil[0] + math.cos(tomoe_angle) * 8)
                        ty = int(pupil[1] + math.sin(tomoe_angle) * 8)
                        cv2.circle(img, (tx, ty), 3, (0, 0, 0), -1)
                else:
                    # Mangekyo Pinwheel Pattern
                    for m_i in range(3):
                        m_angle = t * 8.0 + m_i * (2 * math.pi / 3.0)
                        mx1 = int(pupil[0] + math.cos(m_angle) * 12)
                        my1 = int(pupil[1] + math.sin(m_angle) * 12)
                        cv2.line(img, pupil, (mx1, my1), (0, 0, 0), 2)

                if random.random() < 0.2:
                    particles.emit_fire_particles(pupil[0], pupil[1], count=1)

            eye_title = "MANGEKYO SHARINGAN" if fx_mode == "MANGEKYO" else "3-TOMOE SHARINGAN"
            cv2.putText(img, eye_title, (p_left_pupil[0] - 60, p_left_pupil[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            return

        if fx_mode == "FACE DOTS":
            for idx, lm in enumerate(lms):
                px, py = int(lm.x * w), int(lm.y * h)
                dot_col = (203, 19, 255) if idx in LIP_INDICES else (255, 230, 0) if idx in EYE_INDICES else (0, 230, 255) if idx in EYEBROW_INDICES else (50, 255, 50)
                cv2.circle(img, (px, py), 2, dot_col, -1)
            return

        elif fx_mode == "CYBER MESH":
            overlay = img.copy()
            for chain in [LIPS_OUTER, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, FACE_OVAL]:
                for c_i in range(len(chain) - 1):
                    p1 = (int(lms[chain[c_i]].x * w), int(lms[chain[c_i]].y * h))
                    p2 = (int(lms[chain[c_i + 1]].x * w), int(lms[chain[c_i + 1]].y * h))
                    cv2.line(overlay, p1, p2, (255, 0, 180), 1)
            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            return

        elif fx_mode == "LASER EYES":
            for pupil in [p_left_pupil, p_right_pupil]:
                cv2.circle(img, pupil, 12, (0, 0, 255), -1)
                end_pt = (pupil[0] + int(math.sin(time.time() * 8) * 15), h)
                cv2.line(img, pupil, end_pt, (0, 0, 255), 8)
                cv2.line(img, pupil, end_pt, (0, 215, 255), 4)
                cv2.line(img, pupil, end_pt, (255, 255, 255), 2)
                particles.emit_fire_particles(pupil[0], pupil[1], count=2)


# ----------------- GESTURE DETECTOR -----------------
class GestureRecognizer:
    @classmethod
    def count_fingers(cls, landmarks):
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        thumb_up = landmarks[4].y < landmarks[3].y
        count = sum([index_up, middle_up, ring_up, pinky_up])
        if thumb_up and (index_up or middle_up):
            count += 1
        return count, index_up, middle_up, ring_up, pinky_up, thumb_up

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

        if index_up and not middle_up and not ring_up and not pinky_up:
            return "FINGER_GUN"

        if index_up and middle_up and not ring_up and not pinky_up:
            return "PEACE"

        if index_up and pinky_up and not middle_up and not ring_up:
            return "ROCK_ON"

        if index_up and middle_up and ring_up and pinky_up:
            return "OPEN_PALM"

        if not index_up and not middle_up and not ring_up and not pinky_up:
            if landmarks[4].y < landmarks[3].y:
                return "THUMBS_UP"
            return "FIST"

        if index_up and not middle_up:
            return "POINTING"

        return "UNKNOWN"


# ----------------- HOLOGRAPHIC "DALE" LETTER PHYSICS ENGINE -----------------
class HoloLetter:
    LETTER_COLORS = {
        'D': (0, 200, 255),
        'A': (255, 200, 0),
        'L': (0, 255, 200),
        'E': (255, 0, 200),
    }

    def __init__(self, x, y, letter, font_scale=3.0, size=52):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 0)
        self.letter = letter
        self.font_scale = font_scale
        self.size = size
        self.color = self.LETTER_COLORS.get(letter, (0, 255, 255))
        self.grabbed = False
        self.grabbed_by = -1
        self.rotation_angle = random.uniform(0, 360)
        self.spin_speed = random.uniform(-1.5, 1.5)
        self.trail = []
        self.mass = 1.0
        self.holo_phase = random.uniform(0, math.pi * 2)

    def update(self, w, h):
        if not self.grabbed:
            self.vy += 0.28
            self.vx *= 0.985
            self.vy *= 0.985
            self.x += self.vx
            self.y += self.vy
            self.spin_speed *= 0.995

            if self.y + self.size >= h - 40:
                self.y = h - 40 - self.size
                self.vy = -abs(self.vy) * 0.72
                self.vx *= 0.88
                self.spin_speed += self.vx * 0.15
                if abs(self.vy) < 1.0:
                    self.vy = 0

            if self.y - self.size <= 65:
                self.y = 65 + self.size
                self.vy = abs(self.vy) * 0.65

            if self.x - self.size <= 0:
                self.x = self.size
                self.vx = abs(self.vx) * 0.7
                self.spin_speed = -self.spin_speed * 0.5
            elif self.x + self.size >= w:
                self.x = w - self.size
                self.vx = -abs(self.vx) * 0.7
                self.spin_speed = -self.spin_speed * 0.5

        self.rotation_angle += self.spin_speed
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 12:
            self.trail.pop(0)

    def contains(self, px, py):
        return math.hypot(self.x - px, self.y - py) < self.size + 20

    def draw(self, img, particles):
        cx, cy = int(self.x), int(self.y)
        t = time.time()
        pulse = 0.85 + 0.15 * math.sin(t * 3.5 + self.holo_phase)
        col = self.color
        bright = (min(255, int(col[0] * pulse)), min(255, int(col[1] * pulse)), min(255, int(col[2] * pulse)))

        for i, tp in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 0.35
            ghost_col = (int(col[0] * alpha), int(col[1] * alpha), int(col[2] * alpha))
            cv2.putText(img, self.letter, (tp[0] - 20, tp[1] + 18),
                        cv2.FONT_HERSHEY_DUPLEX, self.font_scale * 0.45 * alpha,
                        ghost_col, max(1, int(3 * alpha)))

        glow_layer = np.zeros_like(img)
        glow_scale = self.font_scale * 1.05
        text_size = cv2.getTextSize(self.letter, cv2.FONT_HERSHEY_DUPLEX, glow_scale, 6)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(glow_layer, self.letter, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, glow_scale, bright, 6)
        glow_layer = cv2.GaussianBlur(glow_layer, (25, 25), 0)
        cv2.addWeighted(img, 1.0, glow_layer, 0.55, 0, img)

        text_size = cv2.getTextSize(self.letter, cv2.FONT_HERSHEY_DUPLEX, self.font_scale, 4)[0]
        tx = cx - text_size[0] // 2
        ty = cy + text_size[1] // 2
        cv2.putText(img, self.letter, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, self.font_scale, (0, 0, 0), 8)
        cv2.putText(img, self.letter, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, self.font_scale, bright, 4)
        inner_col = (min(255, bright[0] + 80), min(255, bright[1] + 80), min(255, bright[2] + 80))
        cv2.putText(img, self.letter, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, self.font_scale, inner_col, 2)

        scan_y_start = cy - text_size[1] // 2 - 5
        scan_y_end = cy + text_size[1] // 2 + 5
        scan_offset = int((t * 40 + self.holo_phase * 20) % 8)
        overlay = img.copy()
        for sy in range(scan_y_start + scan_offset, scan_y_end, 8):
            if 0 < sy < img.shape[0]:
                cv2.line(overlay, (tx - 5, sy), (tx + text_size[0] + 5, sy), (255, 255, 255), 1)
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        hw, hh = text_size[0] // 2 + 12, text_size[1] // 2 + 12
        bracket_col = (min(255, int(col[0] * 0.7)), min(255, int(col[1] * 0.7)), min(255, int(col[2] * 0.7)))
        bk = 10
        for (bx, by, sx, sy_s) in [(cx - hw, cy - hh, 1, 1), (cx + hw, cy - hh, -1, 1),
                                     (cx - hw, cy + hh, 1, -1), (cx + hw, cy + hh, -1, -1)]:
            cv2.line(img, (bx, by), (bx + sx * bk, by), bracket_col, 2)
            cv2.line(img, (bx, by), (bx, by + sy_s * bk), bracket_col, 2)

        if self.grabbed:
            cv2.circle(img, (cx, cy), self.size + 15, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), self.size + 22, (255, 255, 255), 1)
            if random.random() < 0.4:
                particles.emit_sparkles(cx, cy, self.color, count=2, speed=3)


# ----------------- HOLOGRAPHIC PHOTO / CARD OBJECT -----------------
class HoloPhotoCard:
    def __init__(self, x, y, image_path, w=160, h=100):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-2, 0)
        self.pw = w
        self.ph = h
        self.size = max(w, h) // 2 + 10
        self.color = (0, 255, 255)
        self.grabbed = False
        self.grabbed_by = -1
        self.rotation = 0.0
        self.spin_speed = random.uniform(-0.8, 0.8)
        self.trail = []
        self.photo_thumb = None

        if os.path.exists(image_path):
            try:
                raw_img = cv2.imread(image_path)
                if raw_img is not None:
                    self.photo_thumb = cv2.resize(raw_img, (w, h))
            except Exception as e:
                print("Photo load error:", e)

    def update(self, w_screen, h_screen):
        if not self.grabbed:
            self.vy += 0.25
            self.vx *= 0.985
            self.vy *= 0.985
            self.x += self.vx
            self.y += self.vy
            self.spin_speed *= 0.992

            if self.y + self.ph // 2 >= h_screen - 40:
                self.y = h_screen - 40 - self.ph // 2
                self.vy = -abs(self.vy) * 0.70
                self.vx *= 0.88

            if self.y - self.ph // 2 <= 65:
                self.y = 65 + self.ph // 2
                self.vy = abs(self.vy) * 0.65

            if self.x - self.pw // 2 <= 0:
                self.x = self.pw // 2
                self.vx = abs(self.vx) * 0.70
            elif self.x + self.pw // 2 >= w_screen:
                self.x = w_screen - self.pw // 2
                self.vx = -abs(self.vx) * 0.70

        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 10:
            self.trail.pop(0)

    def contains(self, px, py):
        return math.hypot(self.x - px, self.y - py) < self.size + 25

    def draw(self, img, particles):
        cx, cy = int(self.x), int(self.y)
        hw, hh = self.pw // 2, self.ph // 2
        x1, y1 = cx - hw, cy - hh
        x2, y2 = cx + hw, cy + hh

        for i, tp in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail) * 0.3
            cv2.rectangle(img, (tp[0] - hw, tp[1] - hh), (tp[0] + hw, tp[1] + hh),
                          (int(0 * alpha), int(255 * alpha), int(255 * alpha)), 1)

        if self.photo_thumb is not None:
            sy1, sy2 = max(0, y1), min(img.shape[0], y2)
            sx1, sx2 = max(0, x1), min(img.shape[1], x2)
            if sy2 > sy1 and sx2 > sx1:
                thumb_y1 = sy1 - y1
                thumb_y2 = thumb_y1 + (sy2 - sy1)
                thumb_x1 = sx1 - x1
                thumb_x2 = thumb_x1 + (sx2 - sx1)
                cropped_thumb = self.photo_thumb[thumb_y1:thumb_y2, thumb_x1:thumb_x2]
                overlay = img[sy1:sy2, sx1:sx2]
                blended = cv2.addWeighted(overlay, 0.35, cropped_thumb, 0.65, 0)
                img[sy1:sy2, sx1:sx2] = blended
        else:
            overlay = img.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 20, 50), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.rectangle(img, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 0, 180), 1)

        bk = 14
        for (bx, by, sx, sy_s) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(img, (bx, by), (bx + sx * bk, by), (255, 255, 255), 2)
            cv2.line(img, (bx, by), (bx, by + sy_s * bk), (255, 255, 255), 2)

        badge_text = "📸 DALE'S CAPTURE"
        ts = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.rectangle(img, (x1, y1 - 18), (x1 + ts[0] + 12, y1), (20, 20, 30), -1)
        cv2.rectangle(img, (x1, y1 - 18), (x1 + ts[0] + 12, y1), (0, 255, 255), 1)
        cv2.putText(img, badge_text, (x1 + 6, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)


# ----------------- IRON MAN HOLOGRAPHIC WORKSPACE MANAGER -----------------
class IronManWorkspace:
    def __init__(self, w=1280, h=720):
        spacing = w // 5
        center_y = h // 2 + 50
        self.objects = [
            HoloPhotoCard(w // 2, 220, DALE_PHOTO_PATH, w=170, h=105),
            HoloLetter(spacing * 1, center_y, 'D', font_scale=3.2, size=52),
            HoloLetter(spacing * 2, center_y, 'A', font_scale=3.2, size=52),
            HoloLetter(spacing * 3, center_y, 'L', font_scale=3.2, size=52),
            HoloLetter(spacing * 4, center_y, 'E', font_scale=3.2, size=52),
        ]
        self.active_grab = {}
        self.prev_pinch_pts = {}

    def update(self, w, h):
        for obj in self.objects:
            obj.update(w, h)

        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                a = self.objects[i]
                b = self.objects[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy)
                min_dist = a.size + b.size
                if dist < min_dist and dist > 0.1:
                    nx = dx / dist
                    ny = dy / dist
                    overlap = min_dist - dist

                    if not a.grabbed and not b.grabbed:
                        a.x -= nx * overlap * 0.5
                        a.y -= ny * overlap * 0.5
                        b.x += nx * overlap * 0.5
                        b.y += ny * overlap * 0.5
                    elif a.grabbed and not b.grabbed:
                        b.x += nx * overlap
                        b.y += ny * overlap
                    elif b.grabbed and not a.grabbed:
                        a.x -= nx * overlap
                        a.y -= ny * overlap

                    rel_vx = a.vx - b.vx
                    rel_vy = a.vy - b.vy
                    rel_dot = rel_vx * nx + rel_vy * ny
                    if rel_dot > 0:
                        restitution = 0.75
                        impulse = rel_dot * restitution
                        if not a.grabbed:
                            a.vx -= impulse * nx
                            a.vy -= impulse * ny
                        if not b.grabbed:
                            b.vx += impulse * nx
                            b.vy += impulse * ny

    def try_grab(self, hand_idx, px, py):
        if hand_idx in self.active_grab:
            return
        best_obj = None
        best_dist = float('inf')
        for obj in self.objects:
            if obj.grabbed:
                continue
            d = math.hypot(obj.x - px, obj.y - py)
            if d < obj.size + 30 and d < best_dist:
                best_dist = d
                best_obj = obj
        if best_obj is not None:
            best_obj.grabbed = True
            best_obj.grabbed_by = hand_idx
            best_obj.vx = 0
            best_obj.vy = 0
            self.active_grab[hand_idx] = best_obj
            self.prev_pinch_pts[hand_idx] = (px, py)

    def drag(self, hand_idx, px, py):
        if hand_idx not in self.active_grab:
            return
        obj = self.active_grab[hand_idx]
        obj.x += (px - obj.x) * 0.55
        obj.y += (py - obj.y) * 0.55
        if hand_idx in self.prev_pinch_pts:
            prev = self.prev_pinch_pts[hand_idx]
            obj.vx = (px - prev[0]) * 0.75
            obj.vy = (py - prev[1]) * 0.75
        self.prev_pinch_pts[hand_idx] = (px, py)

    def release(self, hand_idx, particles):
        if hand_idx not in self.active_grab:
            return
        obj = self.active_grab[hand_idx]
        obj.grabbed = False
        obj.grabbed_by = -1
        particles.add_shockwave(int(obj.x), int(obj.y), obj.color, max_radius=55)
        del self.active_grab[hand_idx]
        if hand_idx in self.prev_pinch_pts:
            del self.prev_pinch_pts[hand_idx]

    def release_all(self, particles):
        for idx in list(self.active_grab.keys()):
            self.release(idx, particles)

    def draw(self, img, particles):
        for obj in self.objects:
            obj.draw(img, particles)

    def draw_tractor_beam(self, img, hand_idx, hx, hy):
        if hand_idx not in self.active_grab:
            return
        obj = self.active_grab[hand_idx]
        ox, oy = int(obj.x), int(obj.y)
        cv2.line(img, (hx, hy), (ox, oy), (0, 255, 255), 5)
        cv2.line(img, (hx, hy), (ox, oy), (255, 255, 255), 2)
        dist = math.hypot(ox - hx, oy - hy)
        if dist > 10:
            num_dots = int(dist / 14)
            for i in range(num_dots):
                t = (i + (time.time() * 10) % 1) / max(1, num_dots)
                if t > 1:
                    t -= 1
                dx = int(hx + (ox - hx) * t) + random.randint(-4, 4)
                dy = int(hy + (oy - hy) * t) + random.randint(-4, 4)
                cv2.circle(img, (dx, dy), random.randint(2, 5), obj.color, -1)


# ----------------- SKELETON RENDERING -----------------
def draw_hand_skeleton(img, hand_lms, w, h, particles, hand_idx=0, is_grabbing=False):
    pts = []
    for i in range(21):
        px = int(hand_lms[i].x * w)
        py = int(hand_lms[i].y * h)
        pts.append((px, py))

    for c1, c2 in HAND_CONNECTIONS:
        p1, p2 = pts[c1], pts[c2]
        cv2.line(img, p1, p2, (0, 80, 80), 5)
        bone_col = (0, 255, 255) if not is_grabbing else (0, 200, 255)
        cv2.line(img, p1, p2, bone_col, 2)
        cv2.line(img, p1, p2, (200, 255, 255), 1)

    fingertip_ids = {4, 8, 12, 16, 20}
    joint_ids = {3, 7, 11, 15, 19, 2, 6, 10, 14, 18}
    knuckle_ids = {1, 5, 9, 13, 17}

    for i, pt in enumerate(pts):
        if i == 0:
            pulse_r = int(8 + 3 * math.sin(time.time() * 5))
            cv2.circle(img, pt, pulse_r, (0, 200, 255), 2)
            cv2.circle(img, pt, 4, (255, 255, 255), -1)
        elif i in fingertip_ids:
            cv2.circle(img, pt, 8, (0, 255, 255), 2)
            cv2.circle(img, pt, 5, (255, 255, 255), -1)
            if random.random() < 0.3:
                particles.emit_sparkles(pt[0], pt[1], (0, 255, 255), count=1, speed=1.5)
        elif i in knuckle_ids:
            cv2.circle(img, pt, 6, (0, 220, 220), -1)
            cv2.circle(img, pt, 6, (0, 255, 255), 1)
        elif i in joint_ids:
            cv2.circle(img, pt, 4, (0, 180, 180), -1)
            cv2.circle(img, pt, 4, (0, 255, 255), 1)

    palm_cx = (pts[0][0] + pts[9][0]) // 2
    palm_cy = (pts[0][1] + pts[9][1]) // 2
    if is_grabbing:
        cv2.circle(img, (palm_cx, palm_cy), 18, (0, 255, 0), 2)
        cv2.putText(img, "LOCKED", (palm_cx - 22, palm_cy - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    else:
        cv2.circle(img, (palm_cx, palm_cy), 14, (0, 255, 255), 1)

    hand_label = f"HAND_{hand_idx}"
    cv2.putText(img, hand_label, (pts[0][0] - 20, pts[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)


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


# ----------------- RETROLENS FILTERS -----------------
def apply_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
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
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [147, 20, 255]
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
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
        h_r, w_r = roi.shape[:2]
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

    galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    galaxy_bg[:] = (30, 10, 40)
    for _ in range(800):
        sx = np.random.randint(0, 1920)
        sy = np.random.randint(0, 1080)
        galaxy_bg[sy, sx] = (255, 255, 255)

    print("✨ RETROLENS FX - NARUTO JUTSU, 3D QUANTUM MATTER & STUDIO ✨")

    current_mode_idx = 0     # 0: NARUTO JUTSU
    current_face_fx_idx = 0  # 0: SHARINGAN
    current_filter = 0
    filter_cooldown = 0
    mode_cooldown = 0
    last_timestamp_ms = 0
    flash_timer = 0
    fps_time = time.time()
    fps = 30.0
    exit_requested = False

    particles = ParticleManager()
    iron_workspace = IronManWorkspace()
    matter_sim = QuantumMatterSimulator()
    td_fx = TouchDesignerFX()
    naruto = NarutoJutsuEngine()

    while not exit_requested:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        frame_galaxy = galaxy_bg[:h, :w]
        mode = MODES[current_mode_idx]
        face_fx_mode = FACE_FX_MODES[current_face_fx_idx]

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

        td_fx.update_background(img, mask_person)

        # TouchDesigner Invisibility Cloak Rendering
        if mode == "TD CLOAK":
            img = td_fx.render(img, mask_person, particles)

        # Render Face Tracking & Sharingan Eyes
        if face_results and face_results.face_landmarks and mode != "TD CLOAK":
            FaceFXRenderer.render_face_fx(img, face_results.face_landmarks, face_fx_mode, particles)

        # ----------------- TOUCHLESS SPATIAL BUTTONS -----------------
        buttons = []
        buttons.append(SpatialButton("MODE_0", 10, 10, 78, 38, "🍥 JUTSU", border_color=(255, 140, 0)))
        buttons.append(SpatialButton("MODE_1", 90, 10, 68, 38, "⚛️ 3D", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("MODE_2", 160, 10, 70, 38, "👻 CLOAK", border_color=(0, 255, 200)))
        buttons.append(SpatialButton("MODE_3", 232, 10, 68, 38, "🦾 DALE", border_color=(0, 200, 255)))
        buttons.append(SpatialButton("MODE_4", 302, 10, 66, 38, "⚽ BALL", border_color=(255, 200, 0)))
        buttons.append(SpatialButton("MODE_5", 370, 10, 66, 38, "🎨 DRAW", border_color=(255, 0, 128)))
        buttons.append(SpatialButton("MODE_6", 438, 10, 68, 38, "🌀 PORTAL", border_color=(255, 0, 128)))

        # Mode Specific Buttons
        if mode == "NARUTO JUTSU":
            j_lbl = f"JUTSU:{naruto.active_jutsu.split()[0]}"
            buttons.append(SpatialButton("TOGGLE_JUTSU", 510, 10, 100, 38, j_lbl, border_color=(255, 200, 0)))

        elif mode == "3D MATTER":
            state_lbl = f"STATE:{MATTER_STATES[matter_sim.state_idx]}"
            shape_lbl = f"SHAPE:{MATTER_SHAPES[matter_sim.shape_idx]}"
            buttons.append(SpatialButton("TOGGLE_STATE", 510, 10, 88, 38, state_lbl, border_color=(0, 255, 255)))
            buttons.append(SpatialButton("TOGGLE_SHAPE", 600, 10, 90, 38, shape_lbl, border_color=(255, 0, 180)))

        elif mode == "TD CLOAK":
            td_lbl = f"FX:{TD_CLOAK_MODES[td_fx.mode_idx][:9]}"
            buttons.append(SpatialButton("TD_MODE", 510, 10, 100, 38, td_lbl, border_color=(0, 255, 200)))
            buttons.append(SpatialButton("RESET_BG", 612, 10, 90, 38, "RESET BG 📷", border_color=(0, 255, 255)))

        elif mode == "IRON MAN":
            buttons.append(SpatialButton("RESET_HOLO", 510, 10, 86, 38, "🔄 RESET", border_color=(0, 165, 255)))

        # Universal Action Buttons
        face_lbl = f"👁️ {face_fx_mode.split()[0]}"
        buttons.append(SpatialButton("FACE_FX", w - 245, 10, 84, 38, face_lbl, border_color=(0, 0, 255)))
        buttons.append(SpatialButton("PHOTO", w - 158, 10, 76, 38, "📸 SNAP", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("EXIT", w - 80, 10, 72, 38, "❌ EXIT", border_color=(0, 0, 255), hold_time=0.9))

        gesture_names = []
        hand_centers = []
        pts_portal = []
        pointers = []
        pinching_flags = []

        if hand_results.hand_landmarks:
            for h_idx, hand_lms in enumerate(hand_results.hand_landmarks):
                g_type = GestureRecognizer.classify(hand_lms, w, h)
                gesture_names.append(g_type)

                wx, wy = int(hand_lms[0].x * w), int(hand_lms[0].y * h)
                mx, my = int(hand_lms[9].x * w), int(hand_lms[9].y * h)
                hand_center = ((wx + mx) // 2, (wy + my) // 2)
                hand_centers.append(hand_center)

                thumb_pt = (int(hand_lms[4].x * w), int(hand_lms[4].y * h))
                index_pt = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                mid_pt = (int(hand_lms[12].x * w), int(hand_lms[12].y * h))
                ring_pt = (int(hand_lms[16].x * w), int(hand_lms[16].y * h))
                pinky_pt = (int(hand_lms[20].x * w), int(hand_lms[20].y * h))
                fingertips = [thumb_pt, index_pt, mid_pt, ring_pt, pinky_pt]

                pointers.append(index_pt)
                pinching_flags.append(g_type == "PINCH")

                # ----------------- NARUTO JUTSU RENDERING -----------------
                if mode == "NARUTO JUTSU":
                    # 1. ⚡ CHIDORI (Fist / Claw Grip)
                    if (g_type == "FIST" or naruto.active_jutsu == "CHIDORI ⚡") and naruto.active_jutsu != "RASENGAN 🌀":
                        naruto.render_chidori(img, hand_center, fingertips, particles)
                    # 2. 🌀 RASENGAN (Open Palm cupped)
                    elif (g_type == "OPEN_PALM" or naruto.active_jutsu == "RASENGAN 🌀") and naruto.active_jutsu != "CHIDORI ⚡":
                        naruto.render_rasengan(img, hand_center, particles)
                    # 3. 🔥 KATON FIREBALL (Tiger Seal / Pointing Gun)
                    elif g_type in ["FINGER_GUN", "POINTING"] or naruto.active_jutsu == "KATON FIRE 🔥":
                        naruto.render_katon_fireball(img, index_pt, particles)
                    # 4. 🟣 RASENSHURIKEN (Peace Sign)
                    elif g_type == "PEACE" or naruto.active_jutsu == "RASENSHURIKEN 🟣":
                        naruto.render_rasenshuriken(img, hand_center, particles)
                    else:
                        # Default chakra gathering
                        cv2.circle(img, hand_center, 18, (255, 200, 0), 2)
                        particles.emit_sparkles(hand_center[0], hand_center[1], (255, 230, 0), count=2, speed=3)

                elif mode in ["3D MATTER", "IRON MAN"]:
                    is_grab = (h_idx in iron_workspace.active_grab) if mode == "IRON MAN" else (matter_sim.locked and h_idx == 0)
                    draw_hand_skeleton(img, hand_lms, w, h, particles, hand_idx=h_idx, is_grabbing=is_grab)

                elif mode != "TD CLOAK":
                    for id_lm in [4, 8, 12, 16, 20]:
                        cx_lm, cy_lm = int(hand_lms[id_lm].x * w), int(hand_lms[id_lm].y * h)
                        cv2.circle(img, (cx_lm, cy_lm), 6, (0, 255, 255), -1)

                # ----------------- IRON MAN PINCH & GRAB -----------------
                if mode == "IRON MAN":
                    pinch_pt_x = (thumb_pt[0] + index_pt[0]) // 2
                    pinch_pt_y = (thumb_pt[1] + index_pt[1]) // 2
                    if g_type == "PINCH":
                        if h_idx in iron_workspace.active_grab:
                            iron_workspace.drag(h_idx, pinch_pt_x, pinch_pt_y)
                            iron_workspace.draw_tractor_beam(img, h_idx, pinch_pt_x, pinch_pt_y)
                        else:
                            iron_workspace.try_grab(h_idx, pinch_pt_x, pinch_pt_y)
                            if h_idx in iron_workspace.active_grab:
                                particles.add_shockwave(pinch_pt_x, pinch_pt_y, (0, 255, 255), max_radius=40)
                    else:
                        if h_idx in iron_workspace.active_grab:
                            iron_workspace.release(h_idx, particles)

            # ----------------- MULTI-HAND JUTSUS -----------------
            if len(hand_centers) >= 2:
                dist_hands = math.hypot(hand_centers[0][0] - hand_centers[1][0], hand_centers[0][1] - hand_centers[1][1])
                mid_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
                mid_y = (hand_centers[0][1] + hand_centers[1][1]) // 2

                if mode == "NARUTO JUTSU" and dist_hands < 220:
                    naruto.render_rasenshuriken(img, (mid_x, mid_y), particles)

                pt_idx0 = (int(hand_results.hand_landmarks[0][8].x * w), int(hand_results.hand_landmarks[0][8].y * h))
                pt_idx1 = (int(hand_results.hand_landmarks[1][8].x * w), int(hand_results.hand_landmarks[1][8].y * h))
                if math.hypot(pt_idx0[0] - pt_idx1[0], pt_idx0[1] - pt_idx1[1]) < 35 and time.time() > mode_cooldown:
                    new_mode_idx = (current_mode_idx + 1) % len(MODES)
                    if MODES[current_mode_idx] == "IRON MAN":
                        iron_workspace.release_all(particles)
                    current_mode_idx = new_mode_idx
                    mode_cooldown = time.time() + 1.0
                    particles.add_shockwave(pt_idx0[0], pt_idx0[1], (255, 140, 0), max_radius=80)
                    particles.emit_fireworks(pt_idx0[0], pt_idx0[1], count=25)

        # ----------------- 3D QUANTUM MATTER SIMULATOR -----------------
        if mode == "3D MATTER":
            matter_sim.update(hand_results.hand_landmarks if hand_results else None, w, h, particles)
            matter_sim.draw(img, particles)

        # ----------------- IRON MAN HOLOGRAPHIC WORKSPACE -----------------
        elif mode == "IRON MAN":
            iron_workspace.update(w, h)
            iron_workspace.draw(img, particles)
            if not hand_results.hand_landmarks:
                iron_workspace.release_all(particles)

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
                if btn.bid.startswith("MODE_") and btn.bid[5:].isdigit():
                    new_idx = int(btn.bid[5:])
                    if MODES[current_mode_idx] == "IRON MAN":
                        iron_workspace.release_all(particles)
                    current_mode_idx = new_idx
                elif btn.bid == "TOGGLE_JUTSU":
                    cur_j_idx = NARUTO_JUTSUS.index(naruto.active_jutsu)
                    naruto.active_jutsu = NARUTO_JUTSUS[(cur_j_idx + 1) % len(NARUTO_JUTSUS)]
                    particles.emit_fireworks(w // 2, h // 2, count=30)
                elif btn.bid == "TOGGLE_STATE":
                    matter_sim.state_idx = (matter_sim.state_idx + 1) % len(MATTER_STATES)
                elif btn.bid == "TOGGLE_SHAPE":
                    matter_sim.next_shape(particles)
                elif btn.bid == "TD_MODE":
                    td_fx.next_mode(particles)
                elif btn.bid == "RESET_BG":
                    td_fx.bg_frame = img.astype(np.float32)
                    particles.emit_fireworks(w // 2, h // 2, count=30)
                elif btn.bid == "FACE_FX":
                    current_face_fx_idx = (current_face_fx_idx + 1) % len(FACE_FX_MODES)
                elif btn.bid == "RESET_HOLO":
                    iron_workspace.release_all(particles)
                    iron_workspace = IronManWorkspace(w, h)
                    particles.emit_fireworks(w // 2, h // 2, count=25)
                elif btn.bid == "PHOTO":
                    fname = f"retrolens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fpath = os.path.join(SCREENSHOTS_DIR, fname)
                    cv2.imwrite(fpath, img)
                    print(f"📸 Photo saved: {fpath}")
                    flash_timer = 4
                elif btn.bid == "EXIT":
                    exit_requested = True

        # Draw Interactive Cursor Reticles
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
        cv2.line(img, (0, 56), (w, 56), (255, 140, 0), 2)

        for btn in buttons:
            is_active = False
            if btn.bid.startswith("MODE_") and btn.bid[5:].isdigit():
                is_active = int(btn.bid[5:]) == current_mode_idx
            elif btn.bid == "FACE_FX" and face_fx_mode in ["SHARINGAN", "MANGEKYO"]:
                is_active = True
            btn.draw(img, is_active=is_active)

        # Bottom Bar & HUD
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        if mode == "NARUTO JUTSU":
            status_text = "🍥 NARUTO JUTSUS: ✊ Fist=Chidori ⚡ | 🖐️ Palm=Rasengan 🌀 | 👉 Point=Katon Fire 🔥 | ✌️ Peace=Rasenshuriken 🟣"
        else:
            status_text = f"MODE: {mode}  |  EYES: {face_fx_mode}  |  👉 Pinch or Hover to click buttons!"
        cv2.putText(img, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

        now = time.time()
        fps = 1.0 / max(0.001, now - fps_time)
        fps_time = now
        cv2.putText(img, f"FPS: {int(fps)}", (w - 85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if flash_timer > 0:
            cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
            flash_timer -= 1

        cv2.imshow('RETROLENS FX - Naruto Elemental Jutsu, Sharingan & Studio', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()