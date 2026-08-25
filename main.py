import os
import cv2
import mediapipe as mp
import time
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------
# MediaPipe Task Initializations
# -------------------------------------------------------------
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. Hand Landmarker (Dual-Hand 21 Landmarks)
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=os.path.join(SCRIPT_DIR, 'hand_landmarker.task')),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)

# 2. Face Landmarker (478 3D Facial Landmarks)
face_landmarker = None
try:
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(SCRIPT_DIR, 'face_landmarker.task')),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)
except Exception as e:
    print("Face Landmarker not available:", e)

# 3. Selfie Segmenter (for Portal Background Isolation FX)
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
    print("Segmenter not available:", e)

# -------------------------------------------------------------
# Hand Skeleton Connections & Feature Nodes
# -------------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm Base
]
FINGERTIPS = [4, 8, 12, 16, 20]

# -------------------------------------------------------------
# AI Face Recognition Biometric Anchors (Pure White Matrix)
# -------------------------------------------------------------
BIO_LEFT_PUPIL = 468
BIO_RIGHT_PUPIL = 473
BIO_LEFT_EYE = [33, 160, 158, 133, 153, 144]
BIO_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
BIO_LEFT_EYEBROW = [70, 63, 105, 66, 107]
BIO_RIGHT_EYEBROW = [336, 296, 334, 293, 300]
BIO_NOSE_BRIDGE = [168, 6, 197, 195]
BIO_NOSE_TIP = 1
BIO_SUBNASALE = 2
BIO_NOSE_WINGS = [98, 327]
BIO_MOUTH_CORNERS = [61, 291]
BIO_LIPS_OUTER = [61, 37, 0, 267, 291, 314, 17, 84]
BIO_CHIN_TIP = 152
BIO_CHEEKBONES = [123, 352]
BIO_JAWLINE = [234, 132, 58, 172, 150, 152, 379, 397, 288, 361, 454]

# -------------------------------------------------------------
# Interactive Portal Shaders & Filters
# -------------------------------------------------------------
PORTAL_FILTERS = [
    "GALAXY", "CYBER NEON", "THERMAL", "GLITCH", 
    "DUAL-TONE", "SKETCH", "PIXELATE", "INVERT", 
    "SEPIA", "BLUR", "HOLOGRAM BLUE"
]
current_portal_filter_idx = 0
portal_enabled = True

# Pre-render deep space galaxy texture
galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
galaxy_bg[:] = (25, 8, 35)
for _ in range(800):
    galaxy_bg[np.random.randint(0, 1080), np.random.randint(0, 1920)] = (255, 255, 255)
for _ in range(120):
    cv2.circle(galaxy_bg, (np.random.randint(0, 1920), np.random.randint(0, 1080)), np.random.randint(2, 6), (np.random.randint(180, 255), np.random.randint(100, 255), 255), -1)

def apply_portal_filter(roi, filter_name, x=0, y=0, mask_person=None, frame_galaxy=None):
    if filter_name == "CYBER NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [255, 255, 0]
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "GLITCH":
        h_r, w_r = roi.shape[:2]
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [0, 165, 255]
        filtered[mask_c == 0] = [147, 20, 255]
        return filtered
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (max(1, w_r//12), max(1, h_r//12)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (31, 31), 0)
    elif filter_name == "HOLOGRAM BLUE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blue_holo = np.zeros_like(roi)
        blue_holo[:, :, 0] = np.clip(gray * 1.2, 0, 255) # Blue channel
        blue_holo[:, :, 1] = np.clip(gray * 0.8, 0, 255) # Green channel
        blue_holo[:, :, 2] = np.clip(gray * 0.2, 0, 255) # Red channel
        return blue_holo
    elif filter_name == "GALAXY" and mask_person is not None and frame_galaxy is not None:
        bh, bw = roi.shape[:2]
        roi_mask = mask_person[y:y+bh, x:x+bw]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]
        bg_condition = (roi_mask == 0)
        filtered = roi.copy()
        filtered[bg_condition] = roi_galaxy[bg_condition]
        return filtered
    elif filter_name == "GALAXY":
        bh, bw = roi.shape[:2]
        roi_galaxy = frame_galaxy[y:y+bh, x:x+bw]
        return cv2.addWeighted(roi, 0.4, roi_galaxy, 0.6, 0)
    return roi

# -------------------------------------------------------------
# High-Precision Global Coastlines & Geographic Dataset
# -------------------------------------------------------------
def lonlat_to_xyz(lon, lat, r=1.0):
    phi = math.radians(90 - lat)
    theta = math.radians(lon)
    x = r * math.sin(phi) * math.cos(theta)
    y = -r * math.cos(phi)
    z = r * math.sin(phi) * math.sin(theta)
    return np.array([x, y, z], dtype=np.float32)

COASTLINE_COORDS = {
    'NORTH_AMERICA': [
        (-168, 65), (-160, 71), (-140, 70), (-120, 69), (-90, 73), (-80, 65), (-60, 60),
        (-64, 45), (-70, 42), (-75, 35), (-80, 25), (-82, 28), (-90, 30), (-97, 26),
        (-97, 20), (-90, 15), (-83, 10), (-77, 8), (-80, 9), (-85, 12), (-95, 17),
        (-105, 23), (-115, 30), (-124, 40), (-125, 48), (-135, 57), (-150, 60), (-165, 60), (-168, 65)
    ],
    'SOUTH_AMERICA': [
        (-77, 8), (-70, 12), (-60, 10), (-50, 0), (-35, -5), (-38, -15), (-42, -23),
        (-50, -30), (-58, -38), (-65, -45), (-68, -55), (-75, -50), (-72, -40), (-70, -30),
        (-75, -15), (-80, -2), (-77, 8)
    ],
    'AFRICA': [
        (-6, 36), (10, 37), (25, 32), (32, 31), (34, 27), (43, 12), (51, 11),
        (40, -5), (35, -20), (28, -32), (18, -34), (12, -20), (8, 4), (0, 6),
        (-15, 11), (-17, 15), (-15, 25), (-6, 36)
    ],
    'EUROPE': [
        (-9, 38), (-9, 43), (-1, 46), (-5, 48), (2, 51), (8, 55), (10, 58), (15, 65),
        (25, 71), (30, 70), (30, 60), (20, 55), (12, 45), (15, 40), (23, 38), (28, 41),
        (25, 46), (15, 46), (5, 43), (-3, 37), (-9, 38)
    ],
    'ASIA': [
        (35, 32), (40, 40), (50, 40), (55, 25), (60, 25), (68, 24), (72, 18), (80, 10),
        (85, 20), (90, 22), (98, 10), (103, 1), (108, 15), (118, 24), (122, 30),
        (122, 38), (130, 42), (140, 50), (160, 55), (170, 65), (180, 68), (140, 73),
        (100, 76), (70, 73), (60, 68), (55, 55), (45, 45), (35, 32)
    ],
    'AUSTRALIA': [
        (114, -22), (120, -15), (130, -12), (142, -11), (146, -20), (153, -28),
        (150, -37), (140, -38), (130, -32), (115, -34), (114, -22)
    ],
    'JAPAN': [(130, 32), (135, 34), (140, 38), (142, 43), (140, 41), (132, 34), (130, 32)],
    'UK': [(-5, 50), (-1, 51), (1, 53), (-2, 58), (-5, 58), (-4, 55), (-5, 50)],
    'GREENLAND': [(-50, 60), (-40, 62), (-25, 70), (-20, 80), (-40, 83), (-55, 78), (-50, 60)]
}

GLOBAL_CITIES = [
    (-74.0, 40.7, "NYC"),
    (-0.1, 51.5, "LON"),
    (139.7, 35.7, "TYO"),
    (2.3, 48.9, "PAR"),
    (55.3, 25.3, "DXB"),
    (103.8, 1.3, "SIN"),
    (151.2, -33.9, "SYD"),
    (-118.2, 34.0, "LAX"),
    (-46.6, -23.5, "SAO"),
    (31.2, 30.0, "CAI"),
    (121.5, 31.2, "SHA"),
    (120.9, 14.6, "MNL")
]

FLIGHT_ARCS = [
    ((-74.0, 40.7), (-0.1, 51.5)),
    ((-0.1, 51.5), (55.3, 25.3)),
    ((55.3, 25.3), (103.8, 1.3)),
    ((103.8, 1.3), (139.7, 35.7)),
    ((139.7, 35.7), (151.2, -33.9)),
    ((-118.2, 34.0), (139.7, 35.7))
]

def generate_arc_points(p1, p2, num_pts=16):
    v1 = lonlat_to_xyz(p1[0], p1[1])
    v2 = lonlat_to_xyz(p2[0], p2[1])
    pts = []
    for i in range(num_pts + 1):
        t = i / float(num_pts)
        mid = v1 * (1 - t) + v2 * t
        norm = np.linalg.norm(mid)
        if norm > 1e-5:
            alt = 1.0 + 0.20 * math.sin(t * math.pi)
            pts.append((mid / norm) * alt)
    return np.array(pts, dtype=np.float32)

PRECOMPUTED_COASTLINES_3D = {k: np.array([lonlat_to_xyz(lon, lat) for lon, lat in v], dtype=np.float32) for k, v in COASTLINE_COORDS.items()}
PRECOMPUTED_CITIES_3D = [{'name': name, 'pos': lonlat_to_xyz(lon, lat)} for lon, lat, name in GLOBAL_CITIES]
PRECOMPUTED_FLIGHT_ARCS_3D = [generate_arc_points(p1, p2) for p1, p2 in FLIGHT_ARCS]

GRID_LINES_3D = []
for lat_deg in range(-75, 80, 25):
    line_pts = [lonlat_to_xyz(lon, lat_deg) for lon in range(-180, 185, 10)]
    GRID_LINES_3D.append(np.array(line_pts, dtype=np.float32))
for lon_deg in range(-180, 180, 45):
    line_pts = [lonlat_to_xyz(lon_deg, lat) for lat in range(-85, 90, 10)]
    GRID_LINES_3D.append(np.array(line_pts, dtype=np.float32))

# -------------------------------------------------------------
# 3D Math Helper Functions
# -------------------------------------------------------------
def rotate_points_3d(pts_3d, rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    
    R = Rz @ (Ry @ Rx)
    return pts_3d @ R.T

LIGHT_DIR = np.array([0.45, -0.65, 0.60], dtype=np.float32)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

# -------------------------------------------------------------
# TouchDesigner Professional 3D Holographic Object Engine
# -------------------------------------------------------------
class Holographic3DObject:
    def __init__(self, obj_id, shape_type, x, y, size=70, base_color=(0, 220, 255), title=""):
        self.id = obj_id
        self.shape_type = shape_type
        self.x = float(x)
        self.y = float(y)
        self.base_size = float(size)
        self.base_color = base_color
        self.title = title
        
        # Scale & Smooth Physics
        self.user_scale = 1.0
        self.curr_scale = 0.05
        self.is_giant_mode = False

        # 3D Rotation
        self.rx = 0.38 if shape_type == "DOTTED_GLOBE" else np.random.uniform(0, math.pi)
        self.ry = np.random.uniform(0, math.pi)
        self.rz = 0.08
        self.rot_speed_x = 0.010
        self.rot_speed_y = 0.018
        self.rot_speed_z = 0.006

        # Interaction & Kinematics
        self.is_grabbed = False
        self.grab_hand_idx = -1
        self.is_palmed = False
        self.palmed_hand_idx = -1
        self.hovered = False
        self.two_hand_active = False
        self.init_two_hand_dist = 100.0
        self.init_scale = 1.0

        self.vx = 0.0
        self.vy = 0.0
        self.prev_x = float(x)
        self.prev_y = float(y)
        self.anim_phase = np.random.uniform(0, 2 * math.pi)

        self._init_geometry()

    def _init_geometry(self):
        s = self.base_size
        if self.shape_type == "SOLID_CUBE":
            # 4D Hypercube Tesseract Geometry
            self.vertices = np.array([
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]
            ], dtype=np.float32) * (s * 0.65)
            self.inner_core = self.vertices * 0.45
            self.faces = [
                [4, 5, 6, 7], [1, 0, 3, 2], [3, 2, 6, 7],
                [0, 1, 5, 4], [5, 1, 2, 6], [0, 4, 7, 3]
            ]
        elif self.shape_type == "SOLID_OCTAHEDRON":
            # Sacred Merkabah / Star Tetrahedron
            self.vertices = np.array([
                [0, -1.35, 0], [0, 1.35, 0],
                [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]
            ], dtype=np.float32) * (s * 0.70)
            self.faces = [
                [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
                [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5]
            ]
        elif self.shape_type == "SOLID_GEM":
            # Brilliant Cut Quantum Jewel
            verts = [[0, -1.25, 0]]
            for i in range(6):
                ang = i * (2 * math.pi / 6)
                verts.append([math.cos(ang) * 0.9, -0.4, math.sin(ang) * 0.9])
            for i in range(6):
                ang = (i + 0.5) * (2 * math.pi / 6)
                verts.append([math.cos(ang) * 1.2, 0.15, math.sin(ang) * 1.2])
            verts.append([0, 1.45, 0])
            self.vertices = np.array(verts, dtype=np.float32) * (s * 0.62)
            faces = []
            for i in range(6):
                nxt = (i + 1) % 6
                faces.append([0, 1 + i, 1 + nxt])
                faces.append([1 + i, 7 + i, 1 + nxt])
                faces.append([1 + nxt, 7 + i, 7 + nxt])
                faces.append([13, 7 + nxt, 7 + i])
            self.faces = faces
        elif self.shape_type == "SOLID_PRISM":
            verts = []
            for i in range(6):
                ang = i * (2 * math.pi / 6)
                verts.append([math.cos(ang) * 0.95, -0.9, math.sin(ang) * 0.95])
            for i in range(6):
                ang = i * (2 * math.pi / 6)
                verts.append([math.cos(ang) * 0.95, 0.9, math.sin(ang) * 0.95])
            self.vertices = np.array(verts, dtype=np.float32) * (s * 0.6)
            faces = [[0, 5, 4, 3, 2, 1], [6, 7, 8, 9, 10, 11]]
            for i in range(6):
                nxt = (i + 1) % 6
                faces.append([i, nxt, 6 + nxt, 6 + i])
            self.faces = faces
        elif self.shape_type == "DOTTED_GLOBE":
            self.vertices = np.zeros((1, 3), dtype=np.float32)
            self.faces = []

    def get_effective_radius(self):
        return self.base_size * self.curr_scale * 1.15

    def contains(self, px, py):
        radius = max(65.0, self.get_effective_radius() * 1.4)
        return math.hypot(px - self.x, py - self.y) <= radius

    def start_grab(self, hand_idx, px, py):
        self.is_grabbed = True
        self.grab_hand_idx = hand_idx
        self.is_palmed = False
        self.vx = 0.0
        self.vy = 0.0
        self.prev_x = px
        self.prev_y = py

    def move_to(self, px, py):
        self.vx = (px - self.prev_x) * 0.88
        self.vy = (py - self.prev_y) * 0.88
        
        dx = px - self.prev_x
        dy = py - self.prev_y
        self.ry += dx * 0.020
        self.rx += dy * 0.020

        self.prev_x = px
        self.prev_y = py
        self.x = px
        self.y = py

    def release_grab(self):
        self.is_grabbed = False
        self.grab_hand_idx = -1
        self.two_hand_active = False

    def dock_to_palm(self, hand_idx, palm_x, palm_y):
        self.is_palmed = True
        self.palmed_hand_idx = hand_idx
        self.x += (palm_x - self.x) * 0.45
        self.y += (palm_y - self.y) * 0.45
        self.vx = 0.0
        self.vy = 0.0

    def toggle_giant_mode(self, screen_w, screen_h):
        self.is_giant_mode = not self.is_giant_mode
        if self.is_giant_mode:
            self.user_scale = 3.6
            self.x = screen_w / 2.0
            self.y = screen_h / 2.0 + 10
            self.vx = 0.0
            self.vy = 0.0
        else:
            self.user_scale = 1.0

    def zoom_in(self):
        self.user_scale = min(5.0, self.user_scale + 0.4)

    def zoom_out(self):
        self.user_scale = max(0.3, self.user_scale - 0.4)

    def update(self, screen_w, screen_h):
        self.curr_scale += (self.user_scale - self.curr_scale) * 0.22
        self.anim_phase += 0.04

        if self.shape_type == "DOTTED_GLOBE":
            self.ry += 0.020
            self.rx = 0.38
        else:
            self.rx += self.rot_speed_x
            self.ry += self.rot_speed_y
            self.rz += self.rot_speed_z

        if not self.is_grabbed and not self.is_palmed:
            self.x += self.vx
            self.y += self.vy
            self.vx *= 0.90
            self.vy *= 0.90

            radius = self.get_effective_radius()
            margin = radius + 10
            if self.x < margin:
                self.x = margin
                self.vx = abs(self.vx) * 0.7
            elif self.x > screen_w - margin:
                self.x = screen_w - margin
                self.vx = -abs(self.vx) * 0.7

            if self.y < 80 + radius:
                self.y = 80 + radius
                self.vy = abs(self.vy) * 0.7
            elif self.y > screen_h - margin:
                self.y = screen_h - margin
                self.vy = -abs(self.vy) * 0.7

    def draw(self, frame):
        cx, cy = int(self.x), int(self.y)
        fh, fw = frame.shape[:2]
        radius = self.get_effective_radius()

        if cx < -radius*2 or cx > fw + radius*2 or cy < -radius*2 or cy > fh + radius*2:
            return

        # 1. Holographic Kinetic Rings & Aura
        if self.is_palmed:
            cv2.circle(frame, (cx, cy), int(radius * 1.18), (0, 255, 180), 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), int(radius * 1.32), (0, 200, 255), 1, cv2.LINE_AA)
            for i in range(4):
                ang = self.anim_phase * 2.0 + i * (math.pi / 2)
                tx1 = int(cx + math.cos(ang) * radius * 1.25)
                ty1 = int(cy + math.sin(ang) * radius * 1.25)
                tx2 = int(cx + math.cos(ang) * radius * 1.38)
                ty2 = int(cy + math.sin(ang) * radius * 1.38)
                cv2.line(frame, (tx1, ty1), (tx2, ty2), (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "HOLDING IN PALM", (cx - 60, cy - int(radius * 1.35) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 200), 1, cv2.LINE_AA)
        elif self.is_grabbed:
            glow_r = int(radius * 1.14)
            cv2.circle(frame, (cx, cy), glow_r, (0, 165, 255), 2, cv2.LINE_AA)
            status_text = f"2-HAND ZOOM [{self.curr_scale:.1f}x]" if self.two_hand_active else f"DRAGGING [{self.curr_scale:.1f}x]"
            cv2.putText(frame, status_text, (cx - 50, cy - glow_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 200, 255), 1, cv2.LINE_AA)
        elif self.hovered:
            glow_r = int(radius * 1.08)
            cv2.circle(frame, (cx, cy), glow_r, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"PINCH TO GRAB ({self.curr_scale:.1f}x)", (cx - 70, cy - glow_r - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. RENDER 3D HOLOGRAPHIC EARTH GLOBE WITH ACCURATE MAP & FLIGHTS
        # -------------------------------------------------------------
        if self.shape_type == "DOTTED_GLOBE":
            R = self.base_size * 0.95 * self.curr_scale
            r_int = int(R)

            # Atmospheric Corona
            cv2.circle(frame, (cx, cy), r_int + 4, (80, 170, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), r_int + 12, (40, 100, 200), 1, cv2.LINE_AA)
            if self.curr_scale > 1.8:
                cv2.circle(frame, (cx, cy), r_int + 22, (25, 60, 140), 1, cv2.LINE_AA)

            # A. Graticule Grid Lines (Parallels & Meridians)
            for g_line in GRID_LINES_3D:
                rot_grid = rotate_points_3d(g_line * R, self.rx, self.ry, self.rz)
                vis_pts = []
                for pt in rot_grid:
                    if pt[2] > -R * 0.2:
                        vis_pts.append((int(cx + pt[0]), int(cy + pt[1])))
                    else:
                        if len(vis_pts) > 1:
                            cv2.polylines(frame, [np.array(vis_pts, np.int32)], False, (45, 60, 85), 1, cv2.LINE_AA)
                        vis_pts = []
                if len(vis_pts) > 1:
                    cv2.polylines(frame, [np.array(vis_pts, np.int32)], False, (45, 60, 85), 1, cv2.LINE_AA)

            # B. Accurate Continent Coastlines
            for c_name, c_pts in PRECOMPUTED_COASTLINES_3D.items():
                rot_coast = rotate_points_3d(c_pts * R, self.rx, self.ry, self.rz)
                seg = []
                for pt in rot_coast:
                    if pt[2] > 0:
                        seg.append((int(cx + pt[0]), int(cy + pt[1])))
                    else:
                        if len(seg) > 1:
                            cv2.polylines(frame, [np.array(seg, np.int32)], False, (255, 255, 255), 2 if self.curr_scale > 1.8 else 1, cv2.LINE_AA)
                        seg = []
                if len(seg) > 1:
                    cv2.polylines(frame, [np.array(seg, np.int32)], False, (255, 255, 255), 2 if self.curr_scale > 1.8 else 1, cv2.LINE_AA)

            # C. Glowing City Beacons
            for city in PRECOMPUTED_CITIES_3D:
                rot_city = rotate_points_3d(np.array([city['pos'] * R]), self.rx, self.ry, self.rz)[0]
                if rot_city[2] > 0:
                    sx = int(cx + rot_city[0])
                    sy = int(cy + rot_city[1])
                    pulse_r = int(3 + 1.5 * math.sin(self.anim_phase * 3.0))
                    cv2.circle(frame, (sx, sy), pulse_r + 2, (0, 200, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), 2, (255, 255, 255), -1, cv2.LINE_AA)
                    if self.curr_scale > 1.8:
                        cv2.putText(frame, city['name'], (sx + 6, sy + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)

            # D. Dynamic Flight Arcs with Traveling Kinetic Photons
            for arc_pts in PRECOMPUTED_FLIGHT_ARCS_3D:
                rot_arc = rotate_points_3d(arc_pts * R, self.rx, self.ry, self.rz)
                vis_arc = []
                for pt in rot_arc:
                    if pt[2] > -R * 0.1:
                        vis_arc.append((int(cx + pt[0]), int(cy + pt[1])))
                if len(vis_arc) > 1:
                    cv2.polylines(frame, [np.array(vis_arc, np.int32)], False, (0, 220, 255), 1, cv2.LINE_AA)
                    pulse_idx = int((self.anim_phase * 2.0) % len(vis_arc))
                    cv2.circle(frame, vis_arc[pulse_idx], 3, (0, 255, 255), -1, cv2.LINE_AA)

            # E. Orbiting Satellite
            sat_ang = self.anim_phase * 1.5
            sat_pos = np.array([math.cos(sat_ang) * 1.4 * R, math.sin(sat_ang * 0.7) * 0.5 * R, math.sin(sat_ang) * 1.4 * R])
            rot_sat = rotate_points_3d(np.array([sat_pos]), self.rx, self.ry, self.rz)[0]
            if rot_sat[2] > -R * 0.2:
                sx, sy = int(cx + rot_sat[0]), int(cy + rot_sat[1])
                cv2.circle(frame, (sx, sy), 4, (255, 100, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (sx, sy), 7, (255, 200, 255), 1, cv2.LINE_AA)

            globe_label = f"EARTH 3D MESH [{self.curr_scale:.1f}x]" if not self.is_giant_mode else f"HOLOGRAPHIC EARTH 3D [GIANT MODE]"
            cv2.putText(frame, globe_label, (cx - 52, cy + r_int + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. RENDER HIGH-DESIGN 3D SOLID SHAPES
        # -------------------------------------------------------------
        else:
            scaled_verts = self.vertices * self.curr_scale
            rot_v = rotate_points_3d(scaled_verts, self.rx, self.ry, self.rz)

            poly_list = []
            for face in self.faces:
                pts3d = rot_v[face]
                avg_z = np.mean(pts3d[:, 2])

                v1 = pts3d[1] - pts3d[0]
                v2 = pts3d[2] - pts3d[0]
                normal = np.cross(v1, v2)
                norm_len = np.linalg.norm(normal)

                if norm_len > 1e-5:
                    normal = normal / norm_len
                    diffuse = max(0.0, float(np.dot(normal, LIGHT_DIR)))
                    intensity = 0.28 + 0.72 * diffuse
                    poly_list.append((avg_z, face, intensity))

            poly_list.sort(key=lambda item: item[0])

            for avg_z, face, intensity in poly_list:
                pts2d = (rot_v[face, :2] + [cx, cy]).astype(np.int32)
                b_col = self.base_color
                face_color = (
                    int(np.clip(b_col[0] * intensity, 0, 255)),
                    int(np.clip(b_col[1] * intensity, 0, 255)),
                    int(np.clip(b_col[2] * intensity, 0, 255))
                )
                cv2.fillPoly(frame, [pts2d], face_color)
                edge_color = (255, 255, 255) if self.is_grabbed else (220, 240, 255)
                cv2.polylines(frame, [pts2d], True, edge_color, 1, cv2.LINE_AA)

            for v in rot_v:
                vx, vy = int(cx + v[0]), int(cy + v[1])
                cv2.circle(frame, (vx, vy), 3, (255, 255, 255), -1, cv2.LINE_AA)

            # Nested Energy Core for Cube
            if self.shape_type == "SOLID_CUBE":
                rot_inner = rotate_points_3d(self.inner_core * self.curr_scale, self.rx * 1.5, self.ry * 1.5, self.rz * 1.5)
                for f_in in self.faces:
                    p2d_in = (rot_inner[f_in, :2] + [cx, cy]).astype(np.int32)
                    cv2.polylines(frame, [p2d_in], True, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame, f"{self.title} [{self.curr_scale:.1f}x]", (cx - 36, cy + int(self.base_size * self.curr_scale * 0.8) + 18), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


# -------------------------------------------------------------
# Initialize 3D Objects
# -------------------------------------------------------------
draggable_objects = [
    Holographic3DObject("globe_3d", "DOTTED_GLOBE", 240, 260, size=68, base_color=(255, 255, 255), title="EARTH MAP"),
    Holographic3DObject("cube_3d", "SOLID_CUBE", 520, 260, size=55, base_color=(0, 180, 255), title="CYBER TESSERACT"),
    Holographic3DObject("gem_3d", "SOLID_GEM", 780, 260, size=55, base_color=(255, 60, 200), title="STAR MERKABAH")
]

# -------------------------------------------------------------
# Air Drawing & AI Shape Classifier
# -------------------------------------------------------------
air_drawing_mode = False
drawing_canvas_pts = []
shape_detected_banner = ""
shape_banner_timer = 0

def classify_drawn_stroke(pts_list):
    if len(pts_list) < 10:
        return None, None

    contour = np.array(pts_list, dtype=np.int32)
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)

    if peri < 80:
        return None, None

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / max(1, h)
    cx, cy = x + w // 2, y + h // 2

    circularity = 4 * math.pi * area / (peri * peri + 1e-5)
    approx = cv2.approxPolyDP(contour, 0.045 * peri, True)
    num_verts = len(approx)

    if circularity > 0.42 or (0.75 <= aspect_ratio <= 1.35 and area > 1000):
        return "DOTTED_GLOBE", (cx, cy)
    elif num_verts == 3:
        return "SOLID_OCTAHEDRON", (cx, cy)
    elif num_verts == 4:
        if 0.7 <= aspect_ratio <= 1.3:
            return "SOLID_CUBE", (cx, cy)
        else:
            return "SOLID_GEM", (cx, cy)
    elif num_verts in [5, 6]:
        return "SOLID_PRISM", (cx, cy)
    else:
        return "DOTTED_GLOBE", (cx, cy)

# -------------------------------------------------------------
# Touchless Interactive Buttons
# -------------------------------------------------------------
class TouchlessButton:
    def __init__(self, btn_id, text, x, y, w, h, action_name):
        self.id = btn_id
        self.text = text
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.action = action_name
        self.hover_frames = 0
        self.hover_threshold = 10
        self.is_hovered = False

    def contains(self, px, py):
        return (self.x <= px <= self.x + self.w) and (self.y <= py <= self.y + self.h)

    def update(self, pointer_points):
        hovering = any(self.contains(pt[0], pt[1]) for pt in pointer_points)
        triggered = False
        self.is_hovered = hovering
        if hovering:
            self.hover_frames += 1
            if self.hover_frames >= self.hover_threshold:
                triggered = True
                self.hover_frames = 0
        else:
            self.hover_frames = max(0, self.hover_frames - 2)
        return triggered

    def draw(self, frame, custom_color=None):
        overlay = frame.copy()
        bg_col = custom_color if custom_color else ((40, 65, 35) if self.is_hovered else (25, 20, 35))
        cv2.rectangle(overlay, (self.x, self.y), (self.x + self.w, self.y + self.h), bg_col, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        b_col = (0, 255, 120) if self.is_hovered else (100, 110, 140)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), b_col, 1, cv2.LINE_AA)

        if self.hover_frames > 0:
            prog = min(1.0, self.hover_frames / float(self.hover_threshold))
            fill_w = int(self.w * prog)
            cv2.rectangle(frame, (self.x, self.y + self.h - 3), (self.x + fill_w, self.y + self.h), (0, 255, 255), -1)

        cv2.putText(frame, self.text, (self.x + 8, self.y + self.h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

touchless_buttons = [
    TouchlessButton("btn_portal", "🌌 PORTAL: GALAXY", 460, 12, 140, 32, "CYCLE_PORTAL"),
    TouchlessButton("btn_draw", "🎨 DRAW: OFF", 610, 12, 100, 32, "TOGGLE_DRAW"),
    TouchlessButton("btn_giant", "🌍 GIANT", 720, 12, 75, 32, "TOGGLE_GIANT"),
    TouchlessButton("btn_zoom_in", "🔍 +", 805, 12, 50, 32, "ZOOM_IN"),
    TouchlessButton("btn_zoom_out", "🔍 -", 865, 12, 50, 32, "ZOOM_OUT"),
    TouchlessButton("btn_reset", "🔄 RESET", 925, 12, 70, 32, "RESET_ALL")
]

# -------------------------------------------------------------
# Hand Skeleton & Face Marks Draw Functions
# -------------------------------------------------------------
def draw_hand_skeleton(img, hand_landmarks, hand_idx=0):
    h, w = img.shape[:2]
    pts = []
    for lm in hand_landmarks:
        px = int(lm.x * w)
        py = int(lm.y * h)
        pts.append((px, py))

    # Skeleton Lines (Neon Green)
    line_color = (0, 255, 0)
    for p1_idx, p2_idx in HAND_CONNECTIONS:
        if p1_idx < len(pts) and p2_idx < len(pts):
            cv2.line(img, pts[p1_idx], pts[p2_idx], line_color, 2, cv2.LINE_AA)

    # Joint Dots (Red & Green)
    for idx, (px, py) in enumerate(pts):
        if idx in FINGERTIPS:
            cv2.circle(img, (px, py), 7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
        elif idx == 0:
            cv2.circle(img, (px, py), 6, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(img, (px, py), 5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return pts

def draw_face_marks(img, face_landmarks, mode="ALL"):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    if not pts or len(pts) < 478:
        return

    WHITE = (255, 255, 255)
    FAINT_WHITE = (180, 180, 180)

    def draw_path(indices, is_closed=False, color=WHITE, thick=1):
        valid = [pts[i] for i in indices if i < len(pts)]
        if len(valid) > 1:
            cv2.polylines(img, [np.array(valid, np.int32)], is_closed, color, thick, cv2.LINE_AA)

    draw_path(BIO_LEFT_EYEBROW, is_closed=False, color=WHITE, thick=1)
    draw_path(BIO_RIGHT_EYEBROW, is_closed=False, color=WHITE, thick=1)
    draw_path(BIO_LEFT_EYE, is_closed=True, color=WHITE, thick=1)
    draw_path(BIO_RIGHT_EYE, is_closed=True, color=WHITE, thick=1)
    draw_path(BIO_NOSE_BRIDGE, is_closed=False, color=WHITE, thick=1)
    if BIO_NOSE_WINGS[0] < len(pts) and BIO_NOSE_TIP < len(pts) and BIO_NOSE_WINGS[1] < len(pts):
        cv2.line(img, pts[BIO_NOSE_WINGS[0]], pts[BIO_NOSE_TIP], WHITE, 1, cv2.LINE_AA)
        cv2.line(img, pts[BIO_NOSE_TIP], pts[BIO_NOSE_WINGS[1]], WHITE, 1, cv2.LINE_AA)
    draw_path(BIO_LIPS_OUTER, is_closed=True, color=WHITE, thick=1)
    draw_path(BIO_JAWLINE, is_closed=False, color=WHITE, thick=1)

    lp = pts[BIO_LEFT_PUPIL]
    rp = pts[BIO_RIGHT_PUPIL]
    nt = pts[BIO_NOSE_TIP]
    ch = pts[BIO_CHIN_TIP]
    lm = pts[BIO_MOUTH_CORNERS[0]]
    rm = pts[BIO_MOUTH_CORNERS[1]]

    cv2.line(img, lp, rp, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, lp, nt, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, rp, nt, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, nt, lm, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, nt, rm, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, lm, ch, FAINT_WHITE, 1, cv2.LINE_AA)
    cv2.line(img, rm, ch, FAINT_WHITE, 1, cv2.LINE_AA)

    key_recognition_anchors = [
        BIO_LEFT_PUPIL, BIO_RIGHT_PUPIL,
        33, 133, 159, 145, 362, 263, 386, 374,
        70, 105, 107, 336, 334, 300,
        168, 6, 195, BIO_NOSE_TIP, BIO_SUBNASALE, 98, 327,
        61, 291, 0, 17,
        BIO_CHIN_TIP, 123, 352, 10, 151, 234, 454
    ]

    for idx in key_recognition_anchors:
        if idx < len(pts):
            cv2.circle(img, pts[idx], 2, WHITE, -1, cv2.LINE_AA)

    for idx in [BIO_LEFT_PUPIL, BIO_RIGHT_PUPIL, BIO_NOSE_TIP, BIO_CHIN_TIP, 61, 291]:
        if idx < len(pts):
            cv2.circle(img, pts[idx], 5, WHITE, 1, cv2.LINE_AA)

# -------------------------------------------------------------
# Main Application Loop
# -------------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

face_mode = "ALL"
hand_tracking_enabled = True
portal_gesture_cooldown = 0
prev_frame_time = time.time()
fps = 0

print("==================================================")
print("RETROLENS FX: Interactive 3D Objects & Portal System")
print("Controls:")
print("  - [🤏 PINCH] Grab and Drag any 3D object / Earth globe")
print("  - [👐 2-HAND PINCH] Stretch apart to ZOOM BIG / push together to shrink")
print("  - [🖐 PALM] Hold open palm near any object to dock & float it in your hand")
print("  - [🌌 4-FINGER PORTAL] Frame 4 fingertips or tap Thumb-to-Pinky to cycle Portals!")
print("  - [D] Toggle Air Drawing Mode ON / OFF")
print("  - [G] Giant Earth Hologram | [M] Next Portal Filter | [+/-] Zoom | [R] Reset | [Q] Exit")
print("==================================================")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    frame_galaxy = galaxy_bg[:h, :w]

    now = time.time()
    fps = int(1.0 / max(0.001, (now - prev_frame_time)))
    prev_frame_time = now

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = int(time.time() * 1000)

    # 1. Face Landmarker
    face_results = None
    if face_landmarker is not None and face_mode != "OFF":
        try:
            face_results = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception:
            pass

    # 2. Hand Landmarker
    hand_results = None
    try:
        hand_results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception:
        pass

    # 3. Draw Face Marks (Pure White Biometric Matrix)
    if face_results and face_results.face_landmarks:
        for face_lms in face_results.face_landmarks:
            draw_face_marks(img, face_lms, mode=face_mode)

    # 4. Hand Tracking & Simple Gesture Extraction
    active_pinch_points = []
    fingertip_hover_pts = []
    palm_centers = []
    pts_portal = []
    drawing_pointer = None
    cycle_portal_gesture = False
    active_gesture_hint = "PINCH TO GRAB & MOVE | 2-HAND ZOOM" if not air_drawing_mode else "DRAW SHAPE IN THE AIR"

    for obj in draggable_objects:
        obj.hovered = False

    if hand_results and hand_results.hand_landmarks:
        for h_idx, hand_lms in enumerate(hand_results.hand_landmarks):
            pts_2d = draw_hand_skeleton(img, hand_lms, hand_idx=h_idx) if hand_tracking_enabled else [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]

            wrist_pt = pts_2d[0]
            t_pt = pts_2d[4]
            i_pt = pts_2d[8]
            m_pt = pts_2d[12]
            p_pt = pts_2d[20]

            pinch_dist = math.hypot(t_pt[0] - i_pt[0], t_pt[1] - i_pt[1])
            pinch_cx = (t_pt[0] + i_pt[0]) // 2
            pinch_cy = (t_pt[1] + i_pt[1]) // 2
            
            is_pinching = (pinch_dist < 55)

            fingertip_hover_pts.append(i_pt)
            fingertip_hover_pts.append((pinch_cx, pinch_cy))

            # 4-Point Portal Points (Thumb & Index tips of each hand)
            pts_portal.append(t_pt)
            pts_portal.append(i_pt)

            # Palm Center
            mcp_mid = pts_2d[9]
            palm_cx = (wrist_pt[0] + mcp_mid[0]) // 2
            palm_cy = (wrist_pt[1] + mcp_mid[1]) // 2
            is_palm_open = (pinch_dist > 60) and (math.hypot(i_pt[0] - palm_cx, i_pt[1] - palm_cy) > 35)
            if is_palm_open:
                palm_centers.append({'hand_idx': h_idx, 'x': palm_cx, 'y': palm_cy})

            # Cycle filter gesture (Thumb to Pinky touch)
            if math.hypot(t_pt[0] - p_pt[0], t_pt[1] - p_pt[1]) < 42:
                cycle_portal_gesture = True

            if air_drawing_mode and pinch_dist > 45:
                drawing_pointer = i_pt

            active_pinch_points.append({
                "hand_idx": h_idx,
                "cx": pinch_cx,
                "cy": pinch_cy,
                "dist": pinch_dist,
                "is_pinching": is_pinching,
                "thumb_pt": t_pt,
                "index_pt": i_pt
            })

            # Red/Green Pinch Indicator Laser
            if pinch_dist < 80:
                p_color = (0, 255, 0) if is_pinching else (0, 0, 255)
                cv2.line(img, t_pt, i_pt, p_color, 2, cv2.LINE_AA)
                cv2.circle(img, (pinch_cx, pinch_cy), 5, p_color, -1)
                if is_pinching:
                    cv2.circle(img, (pinch_cx, pinch_cy), 12, (0, 255, 0), 1, cv2.LINE_AA)

        if len(hand_results.hand_landmarks) >= 2:
            idx0 = hand_results.hand_landmarks[0][8]
            idx1 = hand_results.hand_landmarks[1][8]
            if math.hypot(idx0.x * w - idx1.x * w, idx0.y * h - idx1.y * h) < 42:
                cycle_portal_gesture = True

    # 5. Cycle Portal Filter on Gesture
    if cycle_portal_gesture:
        if portal_gesture_cooldown == 0:
            current_portal_filter_idx = (current_portal_filter_idx + 1) % len(PORTAL_FILTERS)
            portal_gesture_cooldown = 20
    if portal_gesture_cooldown > 0:
        portal_gesture_cooldown -= 1

    # 6. Interactive 4-Point Portal Window FX
    active_portal_name = PORTAL_FILTERS[current_portal_filter_idx]
    if portal_enabled and len(pts_portal) == 4:
        pts_portal.sort(key=lambda p: p[1])
        top_pts = sorted(pts_portal[:2], key=lambda p: p[0])
        bottom_pts = sorted(pts_portal[2:], key=lambda p: p[0])
        
        poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)
        x_p, y_p, bw_p, bh_p = cv2.boundingRect(poly_pts)
        x_p, y_p = max(0, x_p), max(0, y_p)
        bw_p, bh_p = min(w - x_p, bw_p), min(h - y_p, bh_p)

        if bw_p > 25 and bh_p > 25:
            roi = img[y_p:y_p+bh_p, x_p:x_p+bw_p].copy()
            mask_person = None
            if active_portal_name == "GALAXY" and segmenter is not None:
                seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
                if seg_result.category_mask is not None:
                    mask_person = seg_result.category_mask.numpy_view()
                    if mask_person.shape != (h, w):
                        mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

            filtered_roi = apply_portal_filter(roi, active_portal_name, x_p, y_p, mask_person, frame_galaxy)
            
            mask = np.zeros((bh_p, bw_p), dtype=np.uint8)
            poly_roi = poly_pts - [x_p, y_p]
            cv2.fillPoly(mask, [poly_roi], 255)
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            img[y_p:y_p+bh_p, x_p:x_p+bw_p] = np.where(mask_3ch == 255, filtered_roi, roi)
            cv2.polylines(img, [poly_pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"PORTAL: {active_portal_name}", (top_pts[0][0], max(20, top_pts[0][1] - 10)), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # 7. Air Drawing Processing (Only in Draw Mode)
    if air_drawing_mode:
        if drawing_pointer is not None:
            px, py = drawing_pointer
            drawing_canvas_pts.append((px, py))
            active_gesture_hint = "🎨 DRAWING SHAPE (Circle/Square/Triangle)..."
            cv2.circle(img, (px, py), 8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            if len(drawing_canvas_pts) > 8:
                shape_type, center_pos = classify_drawn_stroke(drawing_canvas_pts)
                if shape_type is not None and center_pos is not None:
                    cx_draw, cy_draw = center_pos
                    new_id = f"shape_{len(draggable_objects) + 1}"
                    color_map = {
                        "DOTTED_GLOBE": (255, 255, 255),
                        "SOLID_CUBE": (0, 180, 255),
                        "SOLID_OCTAHEDRON": (0, 255, 140),
                        "SOLID_GEM": (255, 60, 200),
                        "SOLID_PRISM": (255, 180, 0)
                    }
                    obj_col = color_map.get(shape_type, (0, 220, 255))
                    name_map = {
                        "DOTTED_GLOBE": "EARTH 3D MESH",
                        "SOLID_CUBE": "CYBER TESSERACT",
                        "SOLID_OCTAHEDRON": "3D PYRAMID",
                        "SOLID_GEM": "STAR MERKABAH",
                        "SOLID_PRISM": "3D PRISM"
                    }
                    obj_title = name_map.get(shape_type, "3D SHAPE")
                    new_obj = Holographic3DObject(new_id, shape_type, cx_draw, cy_draw, size=68, base_color=obj_col, title=obj_title)
                    draggable_objects.append(new_obj)
                    shape_detected_banner = f"✨ POPPED OUT: {obj_title}!"
                    shape_banner_timer = 50
                    air_drawing_mode = False
                drawing_canvas_pts = []

    # Render drawn stroke
    if len(drawing_canvas_pts) > 1:
        for i in range(1, len(drawing_canvas_pts)):
            cv2.line(img, drawing_canvas_pts[i - 1], drawing_canvas_pts[i], (0, 255, 255), 3, cv2.LINE_AA)

    # 8. Touchless Action Buttons
    for btn in touchless_buttons:
        if btn.id == "btn_draw":
            btn.text = "🎨 DRAW: ON" if air_drawing_mode else "🎨 DRAW: OFF"
        elif btn.id == "btn_portal":
            btn.text = f"🌌 {active_portal_name[:10]}"

        if btn.update(fingertip_hover_pts):
            if btn.action == "CYCLE_PORTAL":
                current_portal_filter_idx = (current_portal_filter_idx + 1) % len(PORTAL_FILTERS)
            elif btn.action == "TOGGLE_DRAW":
                air_drawing_mode = not air_drawing_mode
                drawing_canvas_pts = []
            elif btn.action == "TOGGLE_GIANT":
                for obj in draggable_objects:
                    if obj.shape_type == "DOTTED_GLOBE":
                        obj.toggle_giant_mode(w, h)
                        draggable_objects.remove(obj)
                        draggable_objects.append(obj)
                        break
            elif btn.action == "ZOOM_IN":
                for obj in reversed(draggable_objects):
                    if obj.hovered or obj.is_grabbed:
                        obj.zoom_in()
                        break
                else:
                    draggable_objects[-1].zoom_in()
            elif btn.action == "ZOOM_OUT":
                for obj in reversed(draggable_objects):
                    if obj.hovered or obj.is_grabbed:
                        obj.zoom_out()
                        break
                else:
                    draggable_objects[-1].zoom_out()
            elif btn.action == "RESET_ALL":
                drawing_canvas_pts = []
                air_drawing_mode = False
                draggable_objects = [
                    Holographic3DObject("globe_3d", "DOTTED_GLOBE", 240, 260, size=68, base_color=(255, 255, 255), title="EARTH MAP"),
                    Holographic3DObject("cube_3d", "SOLID_CUBE", 520, 260, size=55, base_color=(0, 180, 255), title="CYBER TESSERACT"),
                    Holographic3DObject("gem_3d", "SOLID_GEM", 780, 260, size=55, base_color=(255, 60, 200), title="STAR MERKABAH")
                ]

    # 9. Interactive Object Manipulation (Single-Hand Grab, 2-Hand Zoom, Palm Dock)
    if not air_drawing_mode:
        # A. Two-Hand Kinetic Zoom (Both hands pinching)
        two_hand_obj = None
        if len(active_pinch_points) >= 2:
            p0 = active_pinch_points[0]
            p1 = active_pinch_points[1]
            if p0["is_pinching"] and p1["is_pinching"]:
                two_hand_dist = math.hypot(p0["cx"] - p1["cx"], p0["cy"] - p1["cy"])
                mid_x = (p0["cx"] + p1["cx"]) / 2.0
                mid_y = (p0["cy"] + p1["cy"]) / 2.0

                for obj in reversed(draggable_objects):
                    if obj.is_grabbed or obj.contains(p0["cx"], p0["cy"]) or obj.contains(p1["cx"], p1["cy"]) or obj.contains(mid_x, mid_y):
                        two_hand_obj = obj
                        break
                if two_hand_obj is None and len(draggable_objects) > 0:
                    two_hand_obj = draggable_objects[-1]

                if two_hand_obj:
                    active_gesture_hint = f"👐 2-HAND ZOOM [{two_hand_obj.curr_scale:.1f}x]"
                    if not two_hand_obj.two_hand_active:
                        two_hand_obj.two_hand_active = True
                        two_hand_obj.is_grabbed = True
                        two_hand_obj.init_two_hand_dist = max(30.0, two_hand_dist)
                        two_hand_obj.init_scale = two_hand_obj.user_scale
                        draggable_objects.remove(two_hand_obj)
                        draggable_objects.append(two_hand_obj)
                    else:
                        ratio = two_hand_dist / max(30.0, two_hand_obj.init_two_hand_dist)
                        two_hand_obj.user_scale = np.clip(two_hand_obj.init_scale * ratio, 0.3, 5.0)
                        two_hand_obj.move_to(mid_x, mid_y)
                        cv2.line(img, (p0["cx"], p0["cy"]), (p1["cx"], p1["cy"]), (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.circle(img, (int(mid_x), int(mid_y)), 8, (0, 165, 255), -1, cv2.LINE_AA)

        # B. Single-Hand Pinch to Grab & Move
        for p_info in active_pinch_points:
            h_idx = p_info["hand_idx"]
            pcx = p_info["cx"]
            pcy = p_info["cy"]
            is_pinch = p_info["is_pinching"]

            for obj in reversed(draggable_objects):
                if obj.contains(pcx, pcy):
                    obj.hovered = True
                    if is_pinch and not obj.is_grabbed and (two_hand_obj is None):
                        obj.start_grab(h_idx, pcx, pcy)
                        draggable_objects.remove(obj)
                        draggable_objects.append(obj)
                    break

            for obj in draggable_objects:
                if obj.is_grabbed and (obj.grab_hand_idx == h_idx) and not obj.two_hand_active:
                    if is_pinch:
                        active_gesture_hint = f"🤏 MOVING [{obj.title}]"
                        obj.move_to(pcx, pcy)
                        cv2.line(img, (pcx, pcy), (int(obj.x), int(obj.y)), (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        obj.release_grab()

        # C. Palm Docking (Sphere in Hand Levitation)
        for p_data in palm_centers:
            ph_idx = p_data['hand_idx']
            plm_x = p_data['x']
            plm_y = p_data['y']

            for obj in reversed(draggable_objects):
                if not obj.is_grabbed and math.hypot(plm_x - obj.x, plm_y - obj.y) < obj.get_effective_radius() * 1.8:
                    active_gesture_hint = f"🖐 HOLDING [{obj.title}] IN PALM"
                    obj.dock_to_palm(ph_idx, plm_x, plm_y)
                    obj.ry += 0.02
                    break

        for obj in draggable_objects:
            if obj.is_palmed:
                palm_found = any(p['hand_idx'] == obj.palmed_hand_idx and math.hypot(p['x'] - obj.x, p['y'] - obj.y) < obj.get_effective_radius() * 2.2 for p in palm_centers)
                if not palm_found:
                    obj.is_palmed = False

        active_hand_indices = [p["hand_idx"] for p in active_pinch_points]
        for obj in draggable_objects:
            if obj.is_grabbed and (obj.grab_hand_idx not in active_hand_indices):
                obj.release_grab()

    # Update & Draw All 3D Objects & Earth Globe
    for obj in draggable_objects:
        obj.update(w, h)
        obj.draw(img)

    # 10. Draw Touchless Buttons
    for btn in touchless_buttons:
        c_col = (20, 60, 20) if (btn.id == "btn_draw" and air_drawing_mode) else None
        btn.draw(img, custom_color=c_col)

    # 11. Shape Pop-out Banner
    if shape_banner_timer > 0:
        shape_banner_timer -= 1
        overlay_banner = img.copy()
        cv2.rectangle(overlay_banner, (w // 2 - 240, 65), (w // 2 + 240, 105), (20, 40, 20), -1)
        cv2.addWeighted(overlay_banner, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (w // 2 - 240, 65), (w // 2 + 240, 105), (0, 255, 180), 2, cv2.LINE_AA)
        cv2.putText(img, shape_detected_banner, (w // 2 - 225, 92), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # Top HUD Bar
    # -------------------------------------------------------------
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 12, 22), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.line(img, (0, 55), (w, 55), (60, 50, 80), 1)

    cv2.putText(img, "RETROLENS FX", (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"FPS: {fps}", (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 255, 150), 1, cv2.LINE_AA)

    mode_label = "MODE: 🎨 AIR DRAW" if air_drawing_mode else f"PORTAL: 🌌 {active_portal_name}"
    mode_color = (0, 255, 255) if air_drawing_mode else (0, 255, 180)
    cv2.putText(img, mode_label, (180, 24), cv2.FONT_HERSHEY_DUPLEX, 0.46, mode_color, 1, cv2.LINE_AA)
    cv2.putText(img, active_gesture_hint, (180, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    # Show Window
    cv2.imshow('RETROLENS FX - 3D Objects & Interactive Portal FX', img)

    # Keyboard Handlers
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key in [ord('m'), ord('p')]:
        current_portal_filter_idx = (current_portal_filter_idx + 1) % len(PORTAL_FILTERS)
    elif key == ord('d'):
        air_drawing_mode = not air_drawing_mode
        drawing_canvas_pts = []
    elif key == ord('g'):
        for obj in draggable_objects:
            if obj.shape_type == "DOTTED_GLOBE":
                obj.toggle_giant_mode(w, h)
                draggable_objects.remove(obj)
                draggable_objects.append(obj)
                break
    elif key in [ord('+'), ord('=')]:
        for obj in reversed(draggable_objects):
            if obj.hovered or obj.is_grabbed:
                obj.zoom_in()
                break
        else:
            draggable_objects[-1].zoom_in()
    elif key in [ord('-'), ord('_')]:
        for obj in reversed(draggable_objects):
            if obj.hovered or obj.is_grabbed:
                obj.zoom_out()
                break
        else:
            draggable_objects[-1].zoom_out()
    elif key == ord('r'):
        drawing_canvas_pts = []
        air_drawing_mode = False
        draggable_objects = [
            Holographic3DObject("globe_3d", "DOTTED_GLOBE", 240, 260, size=68, base_color=(255, 255, 255), title="EARTH MAP"),
            Holographic3DObject("cube_3d", "SOLID_CUBE", 520, 260, size=55, base_color=(0, 180, 255), title="CYBER TESSERACT"),
            Holographic3DObject("gem_3d", "SOLID_GEM", 780, 260, size=55, base_color=(255, 60, 200), title="STAR MERKABAH")
        ]
    elif key == ord('f'):
        modes = ["ALL", "DOTS", "HUD", "OFF"]
        face_mode = modes[(modes.index(face_mode) + 1) % len(modes)]
    elif key == ord('h'):
        hand_tracking_enabled = not hand_tracking_enabled

cap.release()
cv2.destroyAllWindows()