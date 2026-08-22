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

# ----------------- CONSTANTS & CONFIG -----------------
FILTERS = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"]
MODES = ["BALL GAME", "MAGIC FX", "AIR CANVAS", "RETRO PORTAL"]
FACE_FX_MODES = ["FACE DOTS", "CYBER MESH", "CYBER VISOR", "IRON MAN HUD", "NEON CROWN", "CAT EARS", "LASER EYES", "OFF"]
BALL_TYPES = ["NEON", "FIREBALL", "PLASMA"]

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

# Landmark Groupings for colored Face Mesh
LIP_INDICES = {61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415}
EYE_INDICES = {33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 468, 473}
EYEBROW_INDICES = {70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 336, 296, 334, 293, 300, 276, 283, 282, 295, 285}
NOSE_INDICES = {1, 2, 98, 327, 168, 6, 197, 195, 5, 4, 45, 275, 48, 278, 219, 439, 218, 438}

# Key contour chains for connected mesh lines
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
            self.vy -= 0.35
            self.vx += random.uniform(-0.4, 0.4)
        self.life -= 1.0
        self.size = max(0.5, self.size * 0.96)
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
        if len(self.particles) < 550:
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

    def emit_fire_particles(self, x, y, count=3):
        fire_colors = [(0, 69, 255), (0, 140, 255), (0, 215, 255), (0, 255, 255)]
        for _ in range(count):
            c = random.choice(fire_colors)
            self.add_particle(Particle(
                x + random.uniform(-4, 4), y + random.uniform(-4, 4),
                random.uniform(-0.8, 0.8), random.uniform(-2, -0.5),
                c, random.uniform(3, 7), random.randint(12, 24), decay=0.94, p_type="fire"
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


# ----------------- INTERACTIVE PHYSICS BALL ENGINE -----------------
class InteractiveBall:
    def __init__(self, x=640, y=200):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(random.uniform(-3, 3))
        self.vy = float(-4.0)
        self.radius = 28.0
        self.trail = []
        self.juggles = 0
        self.high_score = 0
        self.last_hit_time = 0.0
        self.ball_type_idx = 0
        self.popup_text = "READY!"
        self.popup_timer = 2.0
        self.hoop_pos = (150, 240)
        self.hoop_radius = 45

    def reset(self, w=1280, h=720):
        self.x = float(w // 2)
        self.y = float(160)
        self.vx = float(random.uniform(-4, 4))
        self.vy = float(-3.0)
        self.juggles = 0
        self.popup_text = "BALL SPAWNED!"
        self.popup_timer = 1.5

    def update(self, w, h, particles):
        gravity = 0.52
        damping = 0.993
        restitution = 0.78

        self.vy += gravity
        self.vx *= damping
        self.vy *= damping

        self.x += self.vx
        self.y += self.vy

        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 12:
            self.trail.pop(0)

        # Floor
        if self.y + self.radius >= h - 35:
            self.y = h - 35 - self.radius
            self.vy = -self.vy * restitution
            self.vx *= 0.85
            if abs(self.vy) > 3:
                particles.emit_sparkles(int(self.x), int(self.y + self.radius), (200, 200, 200), count=4, speed=3)
            if self.juggles > 0:
                self.popup_text = f"DROPPED! Score: {self.juggles}"
                self.popup_timer = 1.5
                self.juggles = 0

        # Ceiling
        if self.y - self.radius <= 60:
            self.y = 60 + self.radius
            self.vy = abs(self.vy) * restitution

        # Left / Right Walls
        if self.x - self.radius <= 0:
            self.x = self.radius
            self.vx = abs(self.vx) * restitution
            particles.emit_sparkles(int(self.x), int(self.y), (0, 255, 255), count=3, speed=3)
        elif self.x + self.radius >= w:
            self.x = w - self.radius
            self.vx = -abs(self.vx) * restitution
            particles.emit_sparkles(int(self.x), int(self.y), (0, 255, 255), count=3, speed=3)

        # Hoop Goal
        hx, hy = self.hoop_pos
        dist_hoop = math.hypot(self.x - hx, self.y - hy)
        if dist_hoop < self.hoop_radius and self.vy > 0:
            self.juggles += 5
            self.high_score = max(self.high_score, self.juggles)
            self.popup_text = "✨ SWISH! +5 PTS! ✨"
            self.popup_timer = 2.0
            particles.emit_fireworks(hx, hy, count=35)
            self.vy = -8.0

        b_type = BALL_TYPES[self.ball_type_idx]
        if b_type == "FIREBALL":
            particles.emit_fire_particles(int(self.x), int(self.y), count=2)
        elif b_type == "PLASMA":
            if random.random() < 0.6:
                particles.emit_sparkles(int(self.x), int(self.y), (255, 0, 255), count=1, speed=2)

    def check_hand_collision(self, hand_landmarks, w, h, particles):
        now = time.time()
        b_type = BALL_TYPES[self.ball_type_idx]

        for hand_lms in hand_landmarks:
            key_nodes = [0, 4, 8, 12, 16, 20, 9]
            for node_id in key_nodes:
                lm = hand_lms[node_id]
                nx, ny = int(lm.x * w), int(lm.y * h)
                node_radius = 16.0 if node_id in [0, 9] else 12.0

                dist = math.hypot(self.x - nx, self.y - ny)
                if dist < self.radius + node_radius:
                    dx = self.x - nx
                    dy = self.y - ny
                    norm = max(0.001, math.hypot(dx, dy))
                    nx_norm, ny_norm = dx / norm, dy / norm

                    impact_speed = random.uniform(12.0, 17.0)
                    self.vx = nx_norm * impact_speed + random.uniform(-2, 2)
                    self.vy = min(-9.0, ny_norm * impact_speed - 4.0)

                    if now - self.last_hit_time > 0.22:
                        self.juggles += 1
                        self.high_score = max(self.high_score, self.juggles)
                        self.last_hit_time = now

                        if self.juggles % 10 == 0:
                            self.popup_text = f"🔥 UNSTOPPABLE! {self.juggles} 🔥"
                            particles.emit_fireworks(int(self.x), int(self.y), count=30)
                        elif self.juggles % 5 == 0:
                            self.popup_text = f"⚡ ON FIRE! {self.juggles} JUGGLES ⚡"
                            particles.add_shockwave(int(self.x), int(self.y), (0, 255, 255), max_radius=70)
                        else:
                            self.popup_text = f"HIT! {self.juggles}"
                        self.popup_timer = 1.0

                        col = (0, 255, 255) if b_type == "NEON" else (0, 140, 255)
                        particles.add_shockwave(int(self.x), int(self.y), col, max_radius=50)
                        particles.emit_sparkles(int(self.x), int(self.y), col, count=8, speed=5)

    def check_face_collision(self, face_landmarks, w, h, particles):
        now = time.time()
        if not face_landmarks:
            return
        lms = face_landmarks[0]
        for f_id in [10, 1]:
            fx, fy = int(lms[f_id].x * w), int(lms[f_id].y * h)
            dist = math.hypot(self.x - fx, self.y - fy)
            if dist < self.radius + 35.0:
                if now - self.last_hit_time > 0.25:
                    self.vy = -13.0
                    self.vx = float((self.x - fx) * 0.4)
                    self.juggles += 2
                    self.high_score = max(self.high_score, self.juggles)
                    self.last_hit_time = now
                    self.popup_text = "⚽ HEADBUTT! +2 PTS"
                    self.popup_timer = 1.2
                    particles.add_shockwave(int(self.x), int(self.y), (255, 200, 0), max_radius=60)
                    particles.emit_sparkles(int(self.x), int(self.y), (255, 255, 255), count=10, speed=6)

    def draw(self, img):
        b_type = BALL_TYPES[self.ball_type_idx]
        pt = (int(self.x), int(self.y))
        r = int(self.radius)

        hx, hy = self.hoop_pos
        cv2.ellipse(img, (hx, hy), (self.hoop_radius, 14), 0, 0, 360, (0, 165, 255), 3)
        cv2.line(img, (hx - self.hoop_radius, hy), (hx - self.hoop_radius // 2, hy + 35), (255, 255, 255), 1)
        cv2.line(img, (hx + self.hoop_radius, hy), (hx + self.hoop_radius // 2, hy + 35), (255, 255, 255), 1)
        cv2.putText(img, "HOOP GOAL (+5)", (hx - 50, hy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        for i, t_pt in enumerate(self.trail):
            alpha = (i + 1) / len(self.trail)
            t_radius = int(r * 0.65 * alpha)
            if b_type == "FIREBALL":
                t_col = (0, int(140 * alpha), int(255 * alpha))
            elif b_type == "PLASMA":
                t_col = (int(255 * alpha), 0, int(200 * alpha))
            else:
                t_col = (int(255 * alpha), int(230 * alpha), 0)
            cv2.circle(img, t_pt, max(1, t_radius), t_col, -1)

        if b_type == "NEON":
            cv2.circle(img, pt, r + 4, (0, 255, 255), 3)
            cv2.circle(img, pt, r, (255, 230, 0), -1)
            cv2.circle(img, (pt[0] - r // 3, pt[1] - r // 3), r // 3, (255, 255, 255), -1)
            cv2.circle(img, pt, r // 2, (0, 180, 200), 2)
        elif b_type == "FIREBALL":
            cv2.circle(img, pt, r + 6, (0, 69, 255), 4)
            cv2.circle(img, pt, r, (0, 165, 255), -1)
            cv2.circle(img, (pt[0] - r // 3, pt[1] - r // 3), r // 3, (255, 255, 200), -1)
        elif b_type == "PLASMA":
            cv2.circle(img, pt, r + 6, (255, 0, 200), 4)
            cv2.circle(img, pt, r, (203, 19, 255), -1)
            cv2.circle(img, (pt[0] - r // 3, pt[1] - r // 3), r // 3, (255, 255, 255), -1)

        if self.popup_timer > 0:
            ts = cv2.getTextSize(self.popup_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0]
            tx = max(20, min(img.shape[1] - ts[0] - 20, int(self.x) - ts[0] // 2))
            ty = max(90, int(self.y) - int(self.radius) - 20)
            cv2.rectangle(img, (tx - 8, ty - ts[1] - 8), (tx + ts[0] + 8, ty + 8), (20, 20, 30), -1)
            cv2.rectangle(img, (tx - 8, ty - ts[1] - 8), (tx + ts[0] + 8, ty + 8), (0, 255, 255), 2)
            cv2.putText(img, self.popup_text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 255), 2)
            self.popup_timer -= 0.04

        sb_x, sb_y = img.shape[1] - 220, 110
        overlay = img.copy()
        cv2.rectangle(overlay, (sb_x, sb_y), (sb_x + 200, sb_y + 75), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (sb_x, sb_y), (sb_x + 200, sb_y + 75), (0, 255, 255), 2)
        cv2.putText(img, f"JUGGLES: {self.juggles}", (sb_x + 12, sb_y + 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(img, f"HIGH SCORE: {self.high_score}", (sb_x + 12, sb_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


# ----------------- SMOOTH AIR DRAWING CANVAS -----------------
class AirCanvas:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_pt = None
        self.brush_color_idx = 0
        self.brush_size = 7
        self.hue_counter = 0
        self.history = []
        self.max_history = 12

    def get_current_color_info(self):
        item = PALETTE[self.brush_color_idx]
        if item["type"] == "rainbow":
            self.hue_counter = (self.hue_counter + 3) % 180
            hsv = np.uint8([[[self.hue_counter, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            return (int(bgr[0]), int(bgr[1]), int(bgr[2])), "rainbow"
        elif item["type"] == "fire":
            return (0, 140, 255), "fire"
        elif item["type"] == "eraser":
            return (0, 0, 0), "eraser"
        return item["color"], "solid"

    def draw_line(self, pt, particles_mgr=None):
        color, b_type = self.get_current_color_info()
        if b_type == "eraser":
            cv2.circle(self.canvas, pt, self.brush_size * 4, (0, 0, 0), -1)
            if self.prev_pt is not None:
                cv2.line(self.canvas, self.prev_pt, pt, (0, 0, 0), self.brush_size * 8)
        else:
            if self.prev_pt is not None:
                dist = math.hypot(pt[0] - self.prev_pt[0], pt[1] - self.prev_pt[1])
                steps = max(1, int(dist / 4))
                for s in range(1, steps + 1):
                    inter_x = int(self.prev_pt[0] + (pt[0] - self.prev_pt[0]) * (s / steps))
                    inter_y = int(self.prev_pt[1] + (pt[1] - self.prev_pt[1]) * (s / steps))
                    cv2.circle(self.canvas, (inter_x, inter_y), self.brush_size // 2, color, -1)
                    if b_type == "fire" and particles_mgr is not None and random.random() < 0.4:
                        particles_mgr.emit_fire_particles(inter_x, inter_y, count=1)
                cv2.line(self.canvas, self.prev_pt, pt, color, self.brush_size)
            else:
                self.save_snapshot()
                cv2.circle(self.canvas, pt, self.brush_size // 2, color, -1)

            if b_type == "fire" and particles_mgr is not None:
                particles_mgr.emit_fire_particles(pt[0], pt[1], count=2)

        self.prev_pt = pt

    def save_snapshot(self):
        if len(self.history) >= self.max_history:
            self.history.pop(0)
        self.history.append(self.canvas.copy())

    def undo(self):
        if self.history:
            self.canvas = self.history.pop()
            self.prev_pt = None

    def clear(self):
        self.save_snapshot()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_pt = None

    def resize(self, w, h):
        self.width, self.height = w, h
        self.canvas = cv2.resize(self.canvas, (w, h))

    def composite(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        glow = cv2.GaussianBlur(self.canvas, (17, 17), 0)
        frame_with_glow = cv2.addWeighted(frame, 1.0, glow, 0.75, 0)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame_with_glow, frame_with_glow, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
        return cv2.add(frame_bg, canvas_fg)


# ----------------- FULL 478-POINT FACE MESH & AR FX ENGINE -----------------
class FaceFXRenderer:
    @staticmethod
    def render_face_fx(img, face_landmarks, fx_mode, particles):
        if not face_landmarks or fx_mode == "OFF":
            return

        h, w = img.shape[:2]
        lms = face_landmarks[0]

        # 1. 🌟 FULL 478-POINT FACE TRACKER DOTS (High Precision Mocap Matrix)
        if fx_mode == "FACE DOTS":
            for idx, lm in enumerate(lms):
                px, py = int(lm.x * w), int(lm.y * h)
                # Feature-based color styling
                if idx in LIP_INDICES:
                    dot_col = (203, 19, 255) # Neon Magenta/Pink
                    dot_r = 3
                elif idx in EYE_INDICES:
                    dot_col = (255, 230, 0)  # Neon Cyan
                    dot_r = 3
                elif idx in EYEBROW_INDICES:
                    dot_col = (0, 230, 255)  # Golden Yellow
                    dot_r = 3
                elif idx in NOSE_INDICES:
                    dot_col = (0, 140, 255)  # Orange
                    dot_r = 3
                else:
                    dot_col = (50, 255, 50)  # Cyber Matrix Green
                    dot_r = 2

                cv2.circle(img, (px, py), dot_r, dot_col, -1)

            # Draw key facial tracking contour lines
            for chain, col in [(LIPS_OUTER, (203, 19, 255)), (LEFT_EYE_CONTOUR, (255, 230, 0)), (RIGHT_EYE_CONTOUR, (255, 230, 0)), (FACE_OVAL, (0, 255, 255))]:
                for c_i in range(len(chain) - 1):
                    p1 = (int(lms[chain[c_i]].x * w), int(lms[chain[c_i]].y * h))
                    p2 = (int(lms[chain[c_i + 1]].x * w), int(lms[chain[c_i + 1]].y * h))
                    cv2.line(img, p1, p2, col, 1)

            # Center target HUD
            p_nose = (int(lms[1].x * w), int(lms[1].y * h))
            cv2.circle(img, p_nose, 8, (255, 255, 255), 1)
            cv2.putText(img, "478-PT FACE MESH", (p_nose[0] - 55, p_nose[1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            return

        # 2. 🌐 CYBER MESH (Sci-fi Wireframe Lattice + Dots)
        elif fx_mode == "CYBER MESH":
            overlay = img.copy()
            # Draw all dots
            for idx, lm in enumerate(lms):
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)

            # Draw mesh triangulation contours
            for chain in [LIPS_OUTER, LEFT_EYE_CONTOUR, RIGHT_EYE_CONTOUR, FACE_OVAL]:
                for c_i in range(len(chain) - 1):
                    p1 = (int(lms[chain[c_i]].x * w), int(lms[chain[c_i]].y * h))
                    p2 = (int(lms[chain[c_i + 1]].x * w), int(lms[chain[c_i + 1]].y * h))
                    cv2.line(overlay, p1, p2, (255, 0, 180), 1)

            cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
            return

        p_left_eye = (int(lms[33].x * w), int(lms[33].y * h))
        p_right_eye = (int(lms[263].x * w), int(lms[263].y * h))
        p_left_pupil = (int(lms[468].x * w), int(lms[468].y * h)) if len(lms) > 468 else p_left_eye
        p_right_pupil = (int(lms[473].x * w), int(lms[473].y * h)) if len(lms) > 473 else p_right_eye
        p_forehead = (int(lms[10].x * w), int(lms[10].y * h))
        p_nose = (int(lms[1].x * w), int(lms[1].y * h))

        dx = p_right_eye[0] - p_left_eye[0]
        dy = p_right_eye[1] - p_left_eye[1]
        eye_dist = max(20, math.hypot(dx, dy))
        angle = math.atan2(dy, dx)

        if fx_mode == "CYBER VISOR":
            mid_eye = ((p_left_eye[0] + p_right_eye[0]) // 2, (p_left_eye[1] + p_right_eye[1]) // 2)
            visor_w = int(eye_dist * 1.5)
            visor_h = int(eye_dist * 0.45)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            pts = np.array([
                [mid_eye[0] - int(visor_w * cos_a / 2 - visor_h * sin_a / 2), mid_eye[1] - int(visor_w * sin_a / 2 + visor_h * cos_a / 2)],
                [mid_eye[0] + int(visor_w * cos_a / 2 - visor_h * sin_a / 2), mid_eye[1] + int(visor_w * sin_a / 2 + visor_h * cos_a / 2)],
                [mid_eye[0] + int(visor_w * cos_a / 2 + visor_h * sin_a / 2), mid_eye[1] + int(visor_w * sin_a / 2 - visor_h * cos_a / 2)],
                [mid_eye[0] - int(visor_w * cos_a / 2 + visor_h * sin_a / 2), mid_eye[1] - int(visor_w * sin_a / 2 - visor_h * cos_a / 2)]
            ], dtype=np.int32)

            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (255, 0, 180))
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)

            scan_y = int(mid_eye[1] + math.sin(time.time() * 6) * (visor_h / 3))
            cv2.line(img, (mid_eye[0] - visor_w // 3, scan_y), (mid_eye[0] + visor_w // 3, scan_y), (255, 255, 255), 1)
            cv2.putText(img, "LOCK ON", (mid_eye[0] - 28, mid_eye[1] - visor_h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        elif fx_mode == "IRON MAN HUD":
            face_w = int(eye_dist * 2.2)
            face_h = int(eye_dist * 2.8)
            cx, cy = p_nose[0], p_nose[1]
            x1, y1 = cx - face_w // 2, cy - face_h // 2
            x2, y2 = cx + face_w // 2, cy + face_h // 2
            corner_len = 25
            for corner in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                px, py, sx, sy = corner
                cv2.line(img, (px, py), (px + sx * corner_len, py), (0, 255, 255), 2)
                cv2.line(img, (px, py), (px, py + sy * corner_len), (0, 255, 255), 2)
            for pupil in [p_left_pupil, p_right_pupil]:
                cv2.circle(img, pupil, 14, (0, 200, 255), 1)
                cv2.circle(img, pupil, 4, (0, 255, 255), -1)
                cv2.line(img, (pupil[0] - 18, pupil[1]), (pupil[0] + 18, pupil[1]), (0, 255, 255), 1)
            cv2.putText(img, "[ TARGET: HUMAN_01 ]", (x1, max(30, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        elif fx_mode == "NEON CROWN":
            halo_y = max(30, p_forehead[1] - int(eye_dist * 0.7))
            halo_cx = p_forehead[0]
            halo_rx = int(eye_dist * 0.85)
            halo_ry = int(eye_dist * 0.25)
            cv2.ellipse(img, (halo_cx, halo_y), (halo_rx, halo_ry), 0, 0, 360, (0, 230, 255), 4)
            cv2.ellipse(img, (halo_cx, halo_y), (halo_rx + 4, halo_ry + 2), 0, 0, 360, (255, 255, 255), 1)
            orb_angle = time.time() * 4.0
            ox = int(halo_cx + halo_rx * math.cos(orb_angle))
            oy = int(halo_y + halo_ry * math.sin(orb_angle))
            cv2.circle(img, (ox, oy), 6, (255, 255, 255), -1)
            particles.emit_sparkles(ox, oy, (0, 255, 255), count=2, speed=2)

        elif fx_mode == "CAT EARS":
            ear_offset_x = int(eye_dist * 0.75)
            ear_top_y = max(20, p_forehead[1] - int(eye_dist * 1.1))
            base_y = p_forehead[1] - int(eye_dist * 0.2)
            l_ear = np.array([[p_forehead[0] - ear_offset_x - 30, base_y], [p_forehead[0] - ear_offset_x, ear_top_y], [p_forehead[0] - 15, base_y - 20]], dtype=np.int32)
            r_ear = np.array([[p_forehead[0] + 15, base_y - 20], [p_forehead[0] + ear_offset_x, ear_top_y], [p_forehead[0] + ear_offset_x + 30, base_y]], dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [l_ear, r_ear], (203, 19, 255))
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            cv2.polylines(img, [l_ear, r_ear], True, (0, 255, 255), 3)
            for side in [-1, 1]:
                wx = p_nose[0] + side * int(eye_dist * 0.6)
                for dy_w in [-10, 0, 10]:
                    cv2.line(img, (wx, p_nose[1] + dy_w), (wx + side * 45, p_nose[1] + dy_w * 2), (255, 255, 255), 2)
            cv2.circle(img, p_nose, 6, (203, 19, 255), -1)

        elif fx_mode == "LASER EYES":
            for pupil in [p_left_pupil, p_right_pupil]:
                cv2.circle(img, pupil, 12, (0, 0, 255), -1)
                cv2.circle(img, pupil, 6, (255, 255, 255), -1)
                end_pt = (pupil[0] + int(math.sin(time.time() * 8) * 15), h)
                cv2.line(img, pupil, end_pt, (0, 0, 255), 8)
                cv2.line(img, pupil, end_pt, (0, 215, 255), 4)
                cv2.line(img, pupil, end_pt, (255, 255, 255), 2)
                particles.emit_fire_particles(pupil[0], pupil[1], count=2)


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
        if math.hypot(tx - ix, ty - iy) < 36:
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


# ----------------- SPATIAL TOUCHLESS UI BUTTONS -----------------
class SpatialButton:
    def __init__(self, bid, x, y, w, h, text, color=(35, 35, 45), border_color=(0, 255, 255), hold_time=0.6):
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

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def update_hover(self, is_hovering, is_pinching=False):
        now = time.time()
        if is_hovering:
            if is_pinching:
                self.hover_start = None
                self.hover_progress = 0.0
                return True
            if self.hover_start is None:
                self.hover_start = now
            elapsed = now - self.hover_start
            self.hover_progress = min(1.0, elapsed / self.hold_time)
            if elapsed >= self.hold_time:
                self.hover_start = None
                self.hover_progress = 0.0
                return True
        else:
            self.hover_start = None
            self.hover_progress = 0.0
        return False

    def draw(self, img, is_active=False):
        bg_col = (60, 60, 85) if self.hover_progress > 0 else self.color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_col, -1)
        border_col = (0, 255, 0) if is_active else self.border_color
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), border_col, 2 if is_active else 1)

        if self.hover_progress > 0:
            fill_w = int(self.w * self.hover_progress)
            cv2.rectangle(img, (self.x, self.y + self.h - 4), (self.x + fill_w, self.y + self.h), (0, 255, 0), -1)

        ts = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        tx = self.x + (self.w - ts[0]) // 2
        ty = self.y + (self.h + ts[1]) // 2
        cv2.putText(img, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


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

    # Galaxy Background for Portal
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

    print("✨ RETROLENS FX - 478-POINT FULL FACE DOTS & BALL STUDIO ✨")

    current_mode_idx = 0
    current_face_fx_idx = 0 # Default to FACE DOTS
    current_filter = 0
    filter_cooldown = 0
    mode_cooldown = 0
    last_timestamp_ms = 0
    flash_timer = 0
    fps_time = time.time()
    fps = 30.0
    exit_requested = False

    air_canvas = AirCanvas()
    particles = ParticleManager()
    ball = InteractiveBall()

    while not exit_requested:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        if air_canvas.width != w or air_canvas.height != h:
            air_canvas.resize(w, h)

        frame_galaxy = galaxy_bg[:h, :w]
        mode = MODES[current_mode_idx]
        face_fx_mode = FACE_FX_MODES[current_face_fx_idx]

        # Monotonic timestamp
        timestamp_ms = time.time_ns() // 1_000_000
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Hand Tracking
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Face Tracking
        face_results = None
        if face_landmarker is not None:
            face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Segmentation
        mask_person = None
        if mode == "RETRO PORTAL" and FILTERS[current_filter] == "GALAXY" and segmenter is not None:
            seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
            if seg_result.category_mask is not None:
                mask_person = seg_result.category_mask.numpy_view()
                if mask_person.shape != (h, w):
                    mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

        # ----------------- RENDER FULL FACE TRACKER DOTS & AR FX -----------------
        if face_results and face_results.face_landmarks:
            FaceFXRenderer.render_face_fx(img, face_results.face_landmarks, face_fx_mode, particles)

        # ----------------- TOUCHLESS SPATIAL BUTTONS -----------------
        buttons = []
        buttons.append(SpatialButton("MODE_0", 12, 12, 68, 36, "BALL", border_color=(255, 200, 0)))
        buttons.append(SpatialButton("MODE_1", 84, 12, 68, 36, "MAGIC", border_color=(255, 0, 128)))
        buttons.append(SpatialButton("MODE_2", 156, 12, 68, 36, "DRAW", border_color=(255, 0, 128)))
        buttons.append(SpatialButton("MODE_3", 228, 12, 68, 36, "PORTAL", border_color=(255, 0, 128)))

        # Face FX Toggle Button
        face_lbl = f"FACE:{face_fx_mode.split()[0]}"
        buttons.append(SpatialButton("FACE_FX", 302, 12, 90, 36, face_lbl, border_color=(0, 255, 255)))

        # Mode Specific Buttons
        if mode == "BALL GAME":
            b_type_lbl = f"TYPE:{BALL_TYPES[ball.ball_type_idx]}"
            buttons.append(SpatialButton("BALL_TYPE", 398, 12, 85, 36, b_type_lbl, border_color=(0, 255, 255)))
            buttons.append(SpatialButton("RESET_BALL", 488, 12, 85, 36, "RESET ⚽", border_color=(0, 140, 255)))

        elif mode == "AIR CANVAS":
            swatch_start_x = 398
            for i, p in enumerate(PALETTE):
                b_col = p["color"] if isinstance(p["color"], tuple) else (255, 255, 255)
                lbl = p["name"][:3]
                buttons.append(SpatialButton(f"COL_{i}", swatch_start_x + (i * 36), 12, 32, 36, lbl, color=(20, 20, 30), border_color=b_col))
            buttons.append(SpatialButton("UNDO", swatch_start_x + (len(PALETTE) * 36) + 6, 12, 50, 36, "UNDO", border_color=(0, 255, 255)))
            buttons.append(SpatialButton("CLEAR", swatch_start_x + (len(PALETTE) * 36) + 58, 12, 52, 36, "CLEAR", border_color=(0, 140, 255)))

        elif mode == "RETRO PORTAL":
            buttons.append(SpatialButton("PREV_FILT", 398, 12, 72, 36, "◀ PREV", border_color=(0, 255, 255)))
            buttons.append(SpatialButton("NEXT_FILT", 474, 12, 72, 36, "NEXT ▶", border_color=(0, 255, 255)))

        # Universal Action Buttons (Right side)
        buttons.append(SpatialButton("PHOTO", w - 165, 12, 75, 36, "📸 SNAP", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("EXIT", w - 85, 12, 72, 36, "❌ EXIT", border_color=(0, 0, 255), hold_time=1.3))

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
                hand_centers.append(((wx + mx) // 2, (wy + my) // 2))

                thumb_pt = (int(hand_lms[4].x * w), int(hand_lms[4].y * h))
                index_pt = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                pinky_pt = (int(hand_lms[20].x * w), int(hand_lms[20].y * h))

                pointers.append(index_pt)
                pinching_flags.append(g_type == "PINCH")

                # Landmarks visualization
                for id_lm in [4, 8, 12, 16, 20]:
                    cx, cy = int(hand_lms[id_lm].x * w), int(hand_lms[id_lm].y * h)
                    cv2.circle(img, (cx, cy), 6, (0, 255, 255), -1)
                    if id_lm in [4, 8]:
                        pts_portal.append([cx, cy])

                # ----------------- MODE SPECIFIC INTERACTIONS -----------------
                if mode == "MAGIC FX":
                    particles.emit_sparkles(index_pt[0], index_pt[1], (0, 255, 255), count=2, speed=2)

                    if g_type == "FINGER_GUN":
                        base_x, base_y = int(hand_lms[5].x * w), int(hand_lms[5].y * h)
                        dx = index_pt[0] - base_x
                        dy = index_pt[1] - base_y
                        norm = math.hypot(dx, dy)
                        if norm > 0:
                            dx, dy = dx / norm, dy / norm
                            end_x = int(index_pt[0] + dx * 900)
                            end_y = int(index_pt[1] + dy * 900)
                            cv2.line(img, index_pt, (end_x, end_y), (255, 0, 200), 8)
                            cv2.line(img, index_pt, (end_x, end_y), (255, 255, 255), 3)
                            cv2.circle(img, index_pt, 12, (255, 255, 255), -1)
                            particles.emit_laser_beam(index_pt[0], index_pt[1], end_x, end_y, (255, 100, 255))

                    elif g_type == "ROCK_ON":
                        steps = 8
                        cur_pt = index_pt
                        for s in range(1, steps + 1):
                            t = s / steps
                            nx = int(index_pt[0] + (pinky_pt[0] - index_pt[0]) * t + random.randint(-12, 12))
                            ny = int(index_pt[1] + (pinky_pt[1] - index_pt[1]) * t + random.randint(-12, 12))
                            cv2.line(img, cur_pt, (nx, ny), (255, 255, 0), 3)
                            cv2.line(img, cur_pt, (nx, ny), (255, 255, 255), 1)
                            cur_pt = (nx, ny)
                        particles.emit_sparkles(index_pt[0], index_pt[1], (255, 255, 0), count=3, speed=4)
                        particles.emit_sparkles(pinky_pt[0], pinky_pt[1], (255, 255, 0), count=3, speed=4)

                    elif g_type == "THUMBS_UP":
                        particles.emit_fireworks(thumb_pt[0], thumb_pt[1], count=14)
                        cv2.putText(img, "AWESOME!", (thumb_pt[0] - 50, thumb_pt[1] - 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

                    elif g_type == "OPEN_PALM":
                        center = hand_centers[h_idx]
                        radius = 65
                        angle_offset = (time.time() * 3) % (2 * math.pi)
                        cv2.circle(img, center, radius, (0, 255, 255), 2)
                        cv2.circle(img, center, radius + 10, (255, 0, 128), 1)
                        for r_i in range(6):
                            theta = angle_offset + (r_i * math.pi / 3)
                            rx = int(center[0] + radius * math.cos(theta))
                            ry = int(center[1] + radius * math.sin(theta))
                            cv2.circle(img, (rx, ry), 5, (0, 255, 255), -1)

                elif mode == "AIR CANVAS":
                    if index_pt[1] > 60:
                        if g_type == "POINTING" or g_type == "PINCH":
                            air_canvas.draw_line(index_pt, particles)
                            particles.emit_sparkles(index_pt[0], index_pt[1], air_canvas.get_current_color_info()[0], count=1, speed=2)
                        elif g_type == "PEACE":
                            air_canvas.prev_pt = None
                            cv2.drawMarker(img, index_pt, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 20, 2)
                        elif g_type == "OPEN_PALM":
                            air_canvas.prev_pt = None
                            cv2.circle(img, hand_centers[h_idx], 50, (0, 0, 255), 2)
                            cv2.circle(air_canvas.canvas, hand_centers[h_idx], 50, (0, 0, 0), -1)
                        else:
                            air_canvas.prev_pt = None
                    else:
                        air_canvas.prev_pt = None

                elif mode == "RETRO PORTAL":
                    if math.hypot(thumb_pt[0] - pinky_pt[0], thumb_pt[1] - pinky_pt[1]) < 40 and time.time() > filter_cooldown:
                        current_filter = (current_filter + 1) % len(FILTERS)
                        filter_cooldown = time.time() + 0.6
                        particles.emit_sparkles(thumb_pt[0], thumb_pt[1], (0, 255, 255), count=15, speed=5)

            # ----------------- MULTI-HAND GESTURES -----------------
            if len(hand_centers) >= 2:
                dist_hands = math.hypot(hand_centers[0][0] - hand_centers[1][0], hand_centers[0][1] - hand_centers[1][1])

                if mode == "MAGIC FX" and dist_hands < 260:
                    mid_x = (hand_centers[0][0] + hand_centers[1][0]) // 2
                    mid_y = (hand_centers[0][1] + hand_centers[1][1]) // 2
                    orb_radius = int(max(20, (260 - dist_hands) / 2))
                    cv2.circle(img, (mid_x, mid_y), orb_radius, (255, 255, 255), -1)
                    cv2.circle(img, (mid_x, mid_y), orb_radius + 8, (255, 200, 0), 4)
                    cv2.circle(img, (mid_x, mid_y), orb_radius + 18, (255, 0, 200), 2)
                    for hc in hand_centers:
                        cv2.line(img, hc, (mid_x + random.randint(-5, 5), mid_y + random.randint(-5, 5)), (0, 255, 255), 2)
                    particles.emit_sparkles(mid_x, mid_y, (0, 255, 255), count=6, speed=6)

                pt_idx0 = (int(hand_results.hand_landmarks[0][8].x * w), int(hand_results.hand_landmarks[0][8].y * h))
                pt_idx1 = (int(hand_results.hand_landmarks[1][8].x * w), int(hand_results.hand_landmarks[1][8].y * h))
                if math.hypot(pt_idx0[0] - pt_idx1[0], pt_idx0[1] - pt_idx1[1]) < 35 and time.time() > mode_cooldown:
                    current_mode_idx = (current_mode_idx + 1) % len(MODES)
                    mode_cooldown = time.time() + 1.0
                    particles.add_shockwave(pt_idx0[0], pt_idx0[1], (255, 0, 255), max_radius=80)
                    particles.emit_sparkles(pt_idx0[0], pt_idx0[1], (255, 0, 255), count=25, speed=6)

                if mode == "RETRO PORTAL" and len(pts_portal) == 4:
                    pts_portal.sort(key=lambda p: p[1])
                    top_pts = sorted(pts_portal[:2], key=lambda p: p[0])
                    bottom_pts = sorted(pts_portal[2:], key=lambda p: p[0])
                    poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)
                    
                    bx, by, bw, bh = cv2.boundingRect(poly_pts)
                    bx, by = max(0, bx), max(0, by)
                    bw, bh = min(w - bx, bw), min(h - by, bh)

                    if bw > 15 and bh > 15:
                        roi = img[by:by+bh, bx:bx+bw].copy()
                        filtered_roi = apply_filter(roi, FILTERS[current_filter], bx, by, mask_person, frame_galaxy)
                        mask = np.zeros((bh, bw), dtype=np.uint8)
                        poly_roi = poly_pts - [bx, by]
                        cv2.fillPoly(mask, [poly_roi], 255)
                        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                        
                        img[by:by+bh, bx:bx+bw] = np.where(mask_3ch == 255, filtered_roi, roi)
                        cv2.polylines(img, [poly_pts], True, (0, 255, 255), 3)

                        for i in range(4):
                            p1, p2 = poly_pts[i], poly_pts[(i + 1) % 4]
                            alpha = random.random()
                            spx = int(p1[0] * alpha + p2[0] * (1 - alpha)) + random.randint(-8, 8)
                            spy = int(p1[1] * alpha + p2[1] * (1 - alpha)) + random.randint(-8, 8)
                            cv2.circle(img, (spx, spy), random.randint(2, 4), (255, 0, 255), -1)

                        cv2.putText(img, f"PORTAL: {FILTERS[current_filter]}", (top_pts[0][0], max(30, top_pts[0][1] - 12)), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        # ----------------- BALL GAME PHYSICS & COLLISIONS -----------------
        if mode == "BALL GAME":
            ball.update(w, h, particles)
            if hand_results.hand_landmarks:
                ball.check_hand_collision(hand_results.hand_landmarks, w, h, particles)
            if face_results and face_results.face_landmarks:
                ball.check_face_collision(face_results.face_landmarks, w, h, particles)
            ball.draw(img)

        # ----------------- BUTTON HOVER & TRIGGER HANDLING -----------------
        for btn in buttons:
            is_hover = False
            is_pinch = False
            for p_idx, pt in enumerate(pointers):
                if btn.contains(pt[0], pt[1]):
                    is_hover = True
                    is_pinch = pinching_flags[p_idx]
                    cv2.circle(img, pt, 16, (0, 255, 255), 2)
                    if btn.hover_progress > 0:
                        angle = int(360 * btn.hover_progress)
                        cv2.ellipse(img, pt, (20, 20), 0, 0, angle, (0, 255, 0), 3)
                    break

            triggered = btn.update_hover(is_hover, is_pinch)
            if triggered:
                particles.add_shockwave(btn.x + btn.w // 2, btn.y + btn.h // 2, (0, 255, 255), max_radius=50)
                if btn.bid == "MODE_0":
                    current_mode_idx = 0
                elif btn.bid == "MODE_1":
                    current_mode_idx = 1
                elif btn.bid == "MODE_2":
                    current_mode_idx = 2
                elif btn.bid == "MODE_3":
                    current_mode_idx = 3
                elif btn.bid == "FACE_FX":
                    current_face_fx_idx = (current_face_fx_idx + 1) % len(FACE_FX_MODES)
                elif btn.bid == "BALL_TYPE":
                    ball.ball_type_idx = (ball.ball_type_idx + 1) % len(BALL_TYPES)
                elif btn.bid == "RESET_BALL":
                    ball.reset(w, h)
                    particles.emit_sparkles(int(ball.x), int(ball.y), (0, 255, 255), count=20, speed=5)
                elif btn.bid.startswith("COL_"):
                    c_idx = int(btn.bid.split("_")[1])
                    air_canvas.brush_color_idx = c_idx
                elif btn.bid == "UNDO":
                    air_canvas.undo()
                elif btn.bid == "CLEAR":
                    air_canvas.clear()
                    particles.emit_fireworks(w // 2, h // 2, count=30)
                elif btn.bid == "PREV_FILT":
                    current_filter = (current_filter - 1) % len(FILTERS)
                elif btn.bid == "NEXT_FILT":
                    current_filter = (current_filter + 1) % len(FILTERS)
                elif btn.bid == "PHOTO":
                    fname = f"retrolens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    fpath = os.path.join(SCREENSHOTS_DIR, fname)
                    cv2.imwrite(fpath, img)
                    print(f"📸 Photo saved: {fpath}")
                    flash_timer = 4
                elif btn.bid == "EXIT":
                    exit_requested = True

        # Composite air canvas if in canvas mode
        if mode == "AIR CANVAS":
            img = air_canvas.composite(img)

        # Update and render particles & shockwaves
        particles.update_and_draw(img)

        # Top HUD bar
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (18, 18, 24), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.line(img, (0, 58), (w, 58), (255, 0, 128), 2)

        # Render buttons
        for btn in buttons:
            is_active = (btn.bid == f"MODE_{current_mode_idx}") or \
                        (btn.bid == f"COL_{air_canvas.brush_color_idx}" and mode == "AIR CANVAS") or \
                        (btn.bid == "FACE_FX" and face_fx_mode != "OFF")
            btn.draw(img, is_active=is_active)

        # Bottom Bar & Status
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        status_text = f"MODE: {mode}  |  FACE FX: {face_fx_mode}  |  Use Hands to Juggle Ball / Draw / Cast Spells!"
        cv2.putText(img, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        now = time.time()
        fps = 1.0 / max(0.001, now - fps_time)
        fps_time = now
        cv2.putText(img, f"FPS: {int(fps)}", (w - 85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if flash_timer > 0:
            cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
            flash_timer -= 1

        cv2.imshow('RETROLENS FX - 478-Point Face Tracker & Ball Studio', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()