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
MODES = ["3D MATTER", "IRON MAN", "BALL GAME", "MAGIC FX", "AIR CANVAS", "RETRO PORTAL"]

# MediaPipe Hand Skeleton Connection Map (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm cross-connections
]

FACE_FX_MODES = ["FACE DOTS", "CYBER MESH", "CYBER VISOR", "IRON MAN HUD", "NEON CROWN", "CAT EARS", "LASER EYES", "OFF"]
BALL_TYPES = ["NEON", "FIREBALL", "PLASMA"]
MATTER_STATES = ["SOLID", "LIQUID", "GAS", "PLASMA"]
MATTER_SHAPES = ["SPHERE", "TORUS", "CUBE", "HEART", "HELIX"]

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


# ----------------- 3D QUANTUM MATTER STATE PARTICLE SIMULATOR -----------------
class QuantumMatterSimulator:
    """Simulates a 3D morphing particle object with Solid, Liquid, Gas, and Plasma physical states."""
    NUM_PARTICLES = 220

    def __init__(self, center_x=640, center_y=360):
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.target_x = float(center_x)
        self.target_y = float(center_y)
        self.radius = 140.0
        self.scale = 1.0
        self.state_idx = 0       # 0: SOLID, 1: LIQUID, 2: GAS, 3: PLASMA
        self.shape_idx = 0       # 0: SPHERE, 1: TORUS, 2: CUBE, 3: HEART, 4: HELIX
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.rot_z = 0.0
        self.spin_x = 0.015
        self.spin_y = 0.022
        self.liquid_wave = 0.0
        self.gas_turbulence = 0.0

        # Generate base 3D coordinates for all shapes
        self.base_shapes = {
            "SPHERE": self._generate_sphere(self.NUM_PARTICLES),
            "TORUS": self._generate_torus(self.NUM_PARTICLES),
            "CUBE": self._generate_cube(self.NUM_PARTICLES),
            "HEART": self._generate_heart(self.NUM_PARTICLES),
            "HELIX": self._generate_helix(self.NUM_PARTICLES)
        }

        # Current particle 3D positions and velocities
        self.particles_pos = np.copy(self.base_shapes["SPHERE"])
        self.particles_vel = np.zeros((self.NUM_PARTICLES, 3), dtype=np.float32)
        self.particle_phases = np.random.uniform(0, math.pi * 2, self.NUM_PARTICLES)

    def _generate_sphere(self, n):
        # Fibonacci sphere lattice
        pts = np.zeros((n, 3), dtype=np.float32)
        phi = math.pi * (math.sqrt(5.0) - 1.0) # golden ratio
        for i in range(n):
            y = 1.0 - (i / float(n - 1)) * 2.0
            radius = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            pts[i] = [math.cos(theta) * radius, y, math.sin(theta) * radius]
        return pts

    def _generate_torus(self, n, R=0.8, r=0.35):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            u = random.uniform(0, math.pi * 2)
            v = random.uniform(0, math.pi * 2)
            x = (R + r * math.cos(v)) * math.cos(u)
            y = (R + r * math.cos(v)) * math.sin(u)
            z = r * math.sin(v)
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
            t = random.uniform(0, math.pi * 2)
            x = 16 * (math.sin(t) ** 3) / 16.0
            y = -(13 * math.cos(t) - 5 * math.cos(2*t) - 2 * math.cos(3*t) - math.cos(4*t)) / 16.0
            z = random.uniform(-0.4, 0.4)
            pts[i] = [x * 0.9, y * 0.9, z]
        return pts

    def _generate_helix(self, n):
        pts = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            strand = 1 if (i % 2 == 0) else -1
            t = (i / float(n)) * math.pi * 6 - math.pi * 3
            pts[i] = [math.cos(t) * 0.6 * strand, t / 4.0, math.sin(t) * 0.6 * strand]
        return pts

    def set_state(self, state_name, particles=None):
        if state_name in MATTER_STATES:
            self.state_idx = MATTER_STATES.index(state_name)
            if particles:
                col = (0, 255, 255) if state_name == "SOLID" else (255, 180, 0) if state_name == "LIQUID" else (255, 0, 255)
                particles.add_shockwave(int(self.center_x), int(self.center_y), col, max_radius=85)

    def next_shape(self, particles=None):
        self.shape_idx = (self.shape_idx + 1) % len(MATTER_SHAPES)
        if particles:
            particles.emit_fireworks(int(self.center_x), int(self.center_y), count=25)

    def update(self, hand_landmarks, w, h, particles):
        current_state = MATTER_STATES[self.state_idx]
        current_shape = MATTER_SHAPES[self.shape_idx]
        base_target = self.base_shapes[current_shape]

        # ----------------- HAND TRACKING & INTERACTION -----------------
        if hand_landmarks:
            lms = hand_landmarks[0]
            # Hand palm center
            hx, hy = int(lms[9].x * w), int(lms[9].y * h)
            self.target_x = hx
            self.target_y = hy

            # Hand rotation estimation
            dx_hand = lms[9].x - lms[0].x
            dy_hand = lms[9].y - lms[0].y
            self.spin_y = dx_hand * 0.08
            self.spin_x = -dy_hand * 0.08

            # Scale from thumb-index pinch distance
            tx, ty = int(lms[4].x * w), int(lms[4].y * h)
            ix, iy = int(lms[8].x * w), int(lms[8].y * h)
            pinch_gap = math.hypot(tx - ix, ty - iy)
            self.scale = max(0.55, min(1.8, pinch_gap / 75.0))
        else:
            self.spin_x = 0.012
            self.spin_y = 0.018

        # Smooth position interpolation
        self.center_x += (self.target_x - self.center_x) * 0.22
        self.center_y += (self.target_y - self.center_y) * 0.22

        # 3D Rotation angles
        self.rot_x += self.spin_x
        self.rot_y += self.spin_y
        self.rot_z += 0.005

        # ----------------- MATTER STATE PHYSICS SIMULATION -----------------
        t = time.time()
        self.liquid_wave += 0.08

        # Rotation matrix computation
        cx, sx = math.cos(self.rot_x), math.sin(self.rot_x)
        cy, sy = math.cos(self.rot_y), math.sin(self.rot_y)
        cz, sz = math.cos(self.rot_z), math.sin(self.rot_z)

        # 1. 🧊 SOLID: Rigid spring tension lattice with diamond structure
        if current_state == "SOLID":
            for i in range(self.NUM_PARTICLES):
                # Rotate base shape
                bx, by, bz = base_target[i]
                # Y-rot
                rx1 = bx * cy + bz * sy
                ry1 = by
                rz1 = -bx * sy + bz * cy
                # X-rot
                rx2 = rx1
                ry2 = ry1 * cx - rz1 * sx
                rz2 = ry1 * sx + rz1 * cx
                # Z-rot
                rx3 = rx2 * cz - ry2 * sz
                ry3 = rx2 * sz + ry2 * cz
                rz3 = rz2

                target_pt = np.array([rx3, ry3, rz3], dtype=np.float32)
                # Strong spring force to lock into rigid geometry
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.35
                self.particles_vel[i] *= 0.8

        # 2. 💧 LIQUID: Viscous fluid dynamics with wave wobble & surface droplet drip
        elif current_state == "LIQUID":
            for i in range(self.NUM_PARTICLES):
                bx, by, bz = base_target[i]
                # Add fluid wave & surface oscillation
                wave = math.sin(self.liquid_wave + self.particle_phases[i]) * 0.18
                fluid_rad = 1.0 + wave
                
                rx1 = bx * cy + bz * sy
                ry1 = by + math.sin(t * 3.0 + bx * 4.0) * 0.15 + 0.12 # slight gravity sag
                rz1 = -bx * sy + bz * cy

                rx2 = rx1 * fluid_rad
                ry2 = ry1 * cx - rz1 * sx
                rz2 = ry1 * sx + rz1 * cx

                target_pt = np.array([rx2, ry2, rz2], dtype=np.float32)
                # Fluid viscosity flow
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.15
                self.particles_vel[i] = np.random.normal(0, 0.02, 3)
                self.particles_pos[i] += self.particles_vel[i]

                # Occasional liquid splash droplet
                if random.random() < 0.015:
                    particles.emit_sparkles(int(self.center_x + rx2 * self.radius * self.scale),
                                            int(self.center_y + ry2 * self.radius * self.scale),
                                            (255, 200, 50), count=1, speed=2)

        # 3. 💨 GAS / VAPOR: High entropy diffusion, Brownian motion, billowing cosmic mist
        elif current_state == "GAS":
            for i in range(self.NUM_PARTICLES):
                bx, by, bz = base_target[i]
                # High expansion
                gas_expand = 1.6 + math.sin(t * 2.0 + self.particle_phases[i]) * 0.45
                
                rx1 = (bx + math.cos(t * 1.5 + i) * 0.4) * gas_expand
                ry1 = (by - math.sin(t * 2.0 + i) * 0.5) * gas_expand - 0.25 # rising smoke
                rz1 = (bz + math.sin(t * 1.5 + i) * 0.4) * gas_expand

                target_pt = np.array([rx1, ry1, rz1], dtype=np.float32)
                # Loose turbulent attraction
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.06
                self.particles_pos[i] += np.random.normal(0, 0.04, 3)

        # 4. ⚡ PLASMA: Superheated high energy vortex with electric discharge
        elif current_state == "PLASMA":
            for i in range(self.NUM_PARTICLES):
                bx, by, bz = base_target[i]
                plasma_spin = t * 6.0 + self.particle_phases[i]
                px = math.cos(plasma_spin) * (0.8 + random.uniform(-0.2, 0.2))
                py = math.sin(plasma_spin) * (0.8 + random.uniform(-0.2, 0.2))
                pz = bz * 1.4 + random.uniform(-0.2, 0.2)

                target_pt = np.array([px, py, pz], dtype=np.float32)
                self.particles_pos[i] += (target_pt - self.particles_pos[i]) * 0.25
                if random.random() < 0.03:
                    particles.emit_fire_particles(int(self.center_x + px * self.radius * self.scale),
                                                  int(self.center_y + py * self.radius * self.scale), count=1)

    def draw(self, img, particles):
        current_state = MATTER_STATES[self.state_idx]
        current_shape = MATTER_SHAPES[self.shape_idx]
        cx, cy = int(self.center_x), int(self.center_y)
        r_current = self.radius * self.scale

        # Project 3D Particles to 2D Screen Space
        projected = []
        for i in range(self.NUM_PARTICLES):
            px, py, pz = self.particles_pos[i]
            # Perspective divide
            camera_dist = 3.5
            factor = camera_dist / (camera_dist + pz)
            sx = int(cx + px * r_current * factor)
            sy = int(cy + py * r_current * factor)
            projected.append((sx, sy, pz, factor, i))

        # Sort particles by Z (Depth Sorting from back to front)
        projected.sort(key=lambda p: p[2])

        # 1. 🧊 SOLID: Draw crystal wireframe interconnects
        if current_state == "SOLID":
            # Draw connecting crystal laser lines between close neighbors
            for i in range(0, min(80, len(projected))):
                p1 = projected[i]
                for j in range(i + 1, min(i + 5, len(projected))):
                    p2 = projected[j]
                    dist_2d = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    if dist_2d < 45 * self.scale:
                        alpha = max(0.1, 1.0 - (dist_2d / (45 * self.scale)))
                        line_col = (int(255 * alpha), int(230 * alpha), int(0 * alpha))
                        cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), line_col, 1)

            # Draw solid crystal points
            for sx, sy, pz, factor, idx in projected:
                depth_alpha = max(0.3, min(1.0, (pz + 1.5) / 3.0))
                dot_r = max(2, int(4 * factor * self.scale))
                col = (int(255 * depth_alpha), int(240 * depth_alpha), int(50 * depth_alpha))
                cv2.circle(img, (sx, sy), dot_r, col, -1)
                cv2.circle(img, (sx, sy), max(1, dot_r // 2), (255, 255, 255), -1)

        # 2. 💧 LIQUID: Draw glowing fluid droplets with specular glints
        elif current_state == "LIQUID":
            for sx, sy, pz, factor, idx in projected:
                depth_alpha = max(0.35, min(1.0, (pz + 1.5) / 3.0))
                dot_r = max(3, int(6 * factor * self.scale))
                # Azure blue fluid color with aqua center
                col = (int(255 * depth_alpha), int(160 * depth_alpha), int(0 * depth_alpha))
                cv2.circle(img, (sx, sy), dot_r, col, -1)
                cv2.circle(img, (sx - 1, sy - 1), max(1, dot_r // 2), (255, 255, 255), -1)

        # 3. 💨 GAS: Draw soft translucent billowing vapor mist
        elif current_state == "GAS":
            for sx, sy, pz, factor, idx in projected:
                depth_alpha = max(0.2, min(0.85, (pz + 1.5) / 3.0))
                dot_r = max(4, int(8 * factor * self.scale))
                col = (int(203 * depth_alpha), int(19 * depth_alpha), int(255 * depth_alpha))
                cv2.circle(img, (sx, sy), dot_r, col, -1)

        # 4. ⚡ PLASMA: Draw electric discharge sparks
        elif current_state == "PLASMA":
            for sx, sy, pz, factor, idx in projected:
                depth_alpha = max(0.3, min(1.0, (pz + 1.5) / 3.0))
                dot_r = max(3, int(5 * factor * self.scale))
                col = (int(0 * depth_alpha), int(140 * depth_alpha), int(255 * depth_alpha))
                cv2.circle(img, (sx, sy), dot_r, col, -1)
                cv2.circle(img, (sx, sy), max(1, dot_r // 2), (255, 255, 255), -1)

        # ----------------- HUD STATUS TELEMETRY -----------------
        hud_x, hud_y = cx - 110, cy - int(r_current) - 40
        overlay = img.copy()
        cv2.rectangle(overlay, (hud_x - 10, hud_y - 20), (hud_x + 230, hud_y + 30), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
        cv2.rectangle(img, (hud_x - 10, hud_y - 20), (hud_x + 230, hud_y + 30), (0, 255, 255), 1)

        # State icon & text
        state_icon = "🧊" if current_state == "SOLID" else "💧" if current_state == "LIQUID" else "💨" if current_state == "GAS" else "⚡"
        status_line = f"{state_icon} STATE: {current_state} | {current_shape}"
        cv2.putText(img, status_line, (hud_x, hud_y), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(img, "✊ FIST=SOLID | 🖐️ PALM=LIQUID | 🤘 ROCK=GAS", (hud_x, hud_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)


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
        if self.y + self.radius >= h - 40:
            self.y = h - 40 - self.radius
            self.vy = -self.vy * restitution
            self.vx *= 0.85
            if abs(self.vy) > 3:
                particles.emit_sparkles(int(self.x), int(self.y + self.radius), (200, 200, 200), count=4, speed=3)
            if self.juggles > 0:
                self.popup_text = f"DROPPED! Score: {self.juggles}"
                self.popup_timer = 1.5
                self.juggles = 0

        # Ceiling
        if self.y - self.radius <= 65:
            self.y = 65 + self.radius
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

        if fx_mode == "FACE DOTS":
            for idx, lm in enumerate(lms):
                px, py = int(lm.x * w), int(lm.y * h)
                if idx in LIP_INDICES:
                    dot_col = (203, 19, 255)
                    dot_r = 3
                elif idx in EYE_INDICES:
                    dot_col = (255, 230, 0)
                    dot_r = 3
                elif idx in EYEBROW_INDICES:
                    dot_col = (0, 230, 255)
                    dot_r = 3
                elif idx in NOSE_INDICES:
                    dot_col = (0, 140, 255)
                    dot_r = 3
                else:
                    dot_col = (50, 255, 50)
                    dot_r = 2

                cv2.circle(img, (px, py), dot_r, dot_col, -1)

            for chain, col in [(LIPS_OUTER, (203, 19, 255)), (LEFT_EYE_CONTOUR, (255, 230, 0)), (RIGHT_EYE_CONTOUR, (255, 230, 0)), (FACE_OVAL, (0, 255, 255))]:
                for c_i in range(len(chain) - 1):
                    p1 = (int(lms[chain[c_i]].x * w), int(lms[chain[c_i]].y * h))
                    p2 = (int(lms[chain[c_i + 1]].x * w), int(lms[chain[c_i + 1]].y * h))
                    cv2.line(img, p1, p2, col, 1)

            p_nose = (int(lms[1].x * w), int(lms[1].y * h))
            cv2.circle(img, p_nose, 8, (255, 255, 255), 1)
            cv2.putText(img, "478-PT FACE MESH", (p_nose[0] - 55, p_nose[1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
            return

        elif fx_mode == "CYBER MESH":
            overlay = img.copy()
            for idx, lm in enumerate(lms):
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)

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
    """A single holographic floating letter with full physics."""
    LETTER_COLORS = {
        'D': (0, 200, 255),   # Amber/Orange - Arc Reactor glow
        'A': (255, 200, 0),   # Cyan blue
        'L': (0, 255, 200),   # Teal / emerald
        'E': (255, 0, 200),   # Magenta / pink
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

            # Floor bounce
            if self.y + self.size >= h - 40:
                self.y = h - 40 - self.size
                self.vy = -abs(self.vy) * 0.72
                self.vx *= 0.88
                self.spin_speed += self.vx * 0.15
                if abs(self.vy) < 1.0:
                    self.vy = 0

            # Ceiling
            if self.y - self.size <= 65:
                self.y = 65 + self.size
                self.vy = abs(self.vy) * 0.65

            # Walls
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
    """A floating draggable holographic picture frame displaying the user's photo."""
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

        if self.grabbed:
            cv2.circle(img, (cx, cy), self.size + 15, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), self.size + 22, (255, 255, 255), 1)
            if random.random() < 0.4:
                particles.emit_sparkles(cx, cy, (0, 255, 255), count=3, speed=3)


# ----------------- IRON MAN HOLOGRAPHIC WORKSPACE MANAGER -----------------
class IronManWorkspace:
    """Manages holographic DALE letters + user photo with full physics."""
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
    """Draw full 21-point Iron Man style holographic hand skeleton."""
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

    print("✨ RETROLENS FX - 3D QUANTUM MATTER, HOLOGRAPH & BALL STUDIO ✨")

    current_mode_idx = 0     # Default to 3D MATTER mode
    current_face_fx_idx = 0  # Default to FACE DOTS
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
    iron_workspace = IronManWorkspace()
    matter_sim = QuantumMatterSimulator()

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
        buttons.append(SpatialButton("MODE_0", 10, 10, 72, 38, "⚛️ MATTER", border_color=(0, 255, 255)))
        buttons.append(SpatialButton("MODE_1", 84, 10, 70, 38, "🦾 DALE", border_color=(0, 200, 255)))
        buttons.append(SpatialButton("MODE_2", 156, 10, 68, 38, "⚽ BALL", border_color=(255, 200, 0)))
        buttons.append(SpatialButton("MODE_3", 226, 10, 68, 38, "⚡ MAGIC", border_color=(255, 0, 128)))
        buttons.append(SpatialButton("MODE_4", 296, 10, 68, 38, "🎨 DRAW", border_color=(255, 0, 128)))
        buttons.append(SpatialButton("MODE_5", 366, 10, 72, 38, "🌀 PORTAL", border_color=(255, 0, 128)))

        # Mode Specific Buttons
        if mode == "3D MATTER":
            state_lbl = f"STATE:{MATTER_STATES[matter_sim.state_idx]}"
            shape_lbl = f"SHAPE:{MATTER_SHAPES[matter_sim.shape_idx]}"
            buttons.append(SpatialButton("TOGGLE_STATE", 442, 10, 92, 38, state_lbl, border_color=(0, 255, 255)))
            buttons.append(SpatialButton("TOGGLE_SHAPE", 536, 10, 94, 38, shape_lbl, border_color=(255, 0, 180)))

        elif mode == "IRON MAN":
            buttons.append(SpatialButton("RESET_HOLO", 442, 10, 88, 38, "🔄 RESET", border_color=(0, 165, 255)))

        elif mode == "BALL GAME":
            b_type_lbl = f"TYPE:{BALL_TYPES[ball.ball_type_idx]}"
            buttons.append(SpatialButton("BALL_TYPE", 442, 10, 86, 38, b_type_lbl, border_color=(0, 255, 255)))
            buttons.append(SpatialButton("RESET_BALL", 530, 10, 86, 38, "RESET ⚽", border_color=(0, 140, 255)))

        elif mode == "AIR CANVAS":
            swatch_start_x = 442
            for i, p in enumerate(PALETTE):
                b_col = p["color"] if isinstance(p["color"], tuple) else (255, 255, 255)
                lbl = p["name"][:3]
                buttons.append(SpatialButton(f"COL_{i}", swatch_start_x + (i * 34), 10, 32, 38, lbl, color=(20, 20, 30), border_color=b_col))
            buttons.append(SpatialButton("UNDO", swatch_start_x + (len(PALETTE) * 34) + 6, 10, 50, 38, "UNDO", border_color=(0, 255, 255)))
            buttons.append(SpatialButton("CLEAR", swatch_start_x + (len(PALETTE) * 34) + 58, 10, 52, 38, "CLEAR", border_color=(0, 140, 255)))

        elif mode == "RETRO PORTAL":
            buttons.append(SpatialButton("PREV_FILT", 442, 10, 72, 38, "◀ PREV", border_color=(0, 255, 255)))
            buttons.append(SpatialButton("NEXT_FILT", 516, 10, 72, 38, "NEXT ▶", border_color=(0, 255, 255)))

        # Universal Action Buttons (Right side)
        face_lbl = f"🎭 {face_fx_mode.split()[0]}"
        buttons.append(SpatialButton("FACE_FX", w - 245, 10, 84, 38, face_lbl, border_color=(0, 255, 255)))
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
                hand_centers.append(((wx + mx) // 2, (wy + my) // 2))

                thumb_pt = (int(hand_lms[4].x * w), int(hand_lms[4].y * h))
                index_pt = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                pinky_pt = (int(hand_lms[20].x * w), int(hand_lms[20].y * h))

                pointers.append(index_pt)
                pinching_flags.append(g_type == "PINCH")

                # Landmarks visualization
                if mode in ["3D MATTER", "IRON MAN"]:
                    is_grab = (h_idx in iron_workspace.active_grab) if mode == "IRON MAN" else (g_type == "FIST")
                    draw_hand_skeleton(img, hand_lms, w, h, particles, hand_idx=h_idx, is_grabbing=is_grab)
                else:
                    for id_lm in [4, 8, 12, 16, 20]:
                        cx_lm, cy_lm = int(hand_lms[id_lm].x * w), int(hand_lms[id_lm].y * h)
                        cv2.circle(img, (cx_lm, cy_lm), 6, (0, 255, 255), -1)

                for id_lm in [4, 8]:
                    cx_lm, cy_lm = int(hand_lms[id_lm].x * w), int(hand_lms[id_lm].y * h)
                    pts_portal.append([cx_lm, cy_lm])

                # ----------------- MODE SPECIFIC INTERACTIONS -----------------
                if mode == "3D MATTER":
                    # Dynamic gesture control of 3D states
                    if g_type == "FIST":
                        if matter_sim.state_idx != 0:
                            matter_sim.set_state("SOLID", particles)
                    elif g_type == "OPEN_PALM":
                        if matter_sim.state_idx != 1:
                            matter_sim.set_state("LIQUID", particles)
                    elif g_type == "ROCK_ON" or g_type == "FINGER_GUN":
                        if matter_sim.state_idx != 2:
                            matter_sim.set_state("GAS", particles)
                    elif g_type == "PEACE" and time.time() > mode_cooldown:
                        matter_sim.next_shape(particles)
                        mode_cooldown = time.time() + 0.8

                elif mode == "IRON MAN":
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

                    pinch_dist = math.hypot(thumb_pt[0] - index_pt[0], thumb_pt[1] - index_pt[1])
                    if pinch_dist < 60:
                        cv2.line(img, thumb_pt, index_pt, (0, 255, 0), 2)
                        cv2.circle(img, (pinch_pt_x, pinch_pt_y), int(pinch_dist / 3) + 4, (0, 255, 0), 2)

                elif mode == "MAGIC FX":
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
                            t_s = s / steps
                            nx = int(index_pt[0] + (pinky_pt[0] - index_pt[0]) * t_s + random.randint(-12, 12))
                            ny = int(index_pt[1] + (pinky_pt[1] - index_pt[1]) * t_s + random.randint(-12, 12))
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
                    new_mode_idx = (current_mode_idx + 1) % len(MODES)
                    if MODES[current_mode_idx] == "IRON MAN":
                        iron_workspace.release_all(particles)
                    current_mode_idx = new_mode_idx
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

        # ----------------- BALL GAME PHYSICS & COLLISIONS -----------------
        elif mode == "BALL GAME":
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
                    break

            triggered = btn.update_hover(is_hover, is_pinch)
            if triggered:
                particles.add_shockwave(btn.x + btn.w // 2, btn.y + btn.h // 2, (0, 255, 255), max_radius=50)
                if btn.bid.startswith("MODE_") and btn.bid[5:].isdigit():
                    new_idx = int(btn.bid[5:])
                    if MODES[current_mode_idx] == "IRON MAN":
                        iron_workspace.release_all(particles)
                    current_mode_idx = new_idx
                elif btn.bid == "TOGGLE_STATE":
                    matter_sim.state_idx = (matter_sim.state_idx + 1) % len(MATTER_STATES)
                    particles.add_shockwave(int(matter_sim.center_x), int(matter_sim.center_y), (0, 255, 255), max_radius=80)
                elif btn.bid == "TOGGLE_SHAPE":
                    matter_sim.next_shape(particles)
                elif btn.bid == "FACE_FX":
                    current_face_fx_idx = (current_face_fx_idx + 1) % len(FACE_FX_MODES)
                elif btn.bid == "BALL_TYPE":
                    ball.ball_type_idx = (ball.ball_type_idx + 1) % len(BALL_TYPES)
                elif btn.bid == "RESET_BALL":
                    ball.reset(w, h)
                    particles.emit_sparkles(int(ball.x), int(ball.y), (0, 255, 255), count=20, speed=5)
                elif btn.bid == "RESET_HOLO":
                    iron_workspace.release_all(particles)
                    iron_workspace = IronManWorkspace(w, h)
                    particles.emit_fireworks(w // 2, h // 2, count=25)
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

        # Draw Futuristic Interactive Cursors at index fingertips
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

        if mode == "AIR CANVAS":
            img = air_canvas.composite(img)

        particles.update_and_draw(img)

        # Top HUD bar background
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (16, 16, 24), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.line(img, (0, 56), (w, 56), (0, 255, 255), 2)

        # Render buttons
        for btn in buttons:
            is_active = False
            if btn.bid.startswith("MODE_") and btn.bid[5:].isdigit():
                is_active = int(btn.bid[5:]) == current_mode_idx
            elif btn.bid == f"COL_{air_canvas.brush_color_idx}" and mode == "AIR CANVAS":
                is_active = True
            elif btn.bid == "FACE_FX" and face_fx_mode != "OFF":
                is_active = True
            btn.draw(img, is_active=is_active)

        # Bottom Bar & Status
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        status_text = f"MODE: {mode}  |  FACE FX: {face_fx_mode}  |  ✊ FIST=Solid  🖐️ PALM=Liquid  🤘 ROCK=Gas  ✌️ PEACE=Shape"
        cv2.putText(img, status_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        now = time.time()
        fps = 1.0 / max(0.001, now - fps_time)
        fps_time = now
        cv2.putText(img, f"FPS: {int(fps)}", (w - 85, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if flash_timer > 0:
            cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
            flash_timer -= 1

        cv2.imshow('RETROLENS FX - 3D Quantum Matter, Holograph & Ball Studio', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()