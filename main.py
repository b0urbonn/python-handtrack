import os
import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np
from collections import deque

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
    print("Segmenter status:", e)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

print("✨ SWIPE LIQUID GLASS / FROSTED CRYSTAL CLOAK STUDIO ACTIVE ✨")
print("👈 SWIPE LEFT: Person turns into CRYSTAL GLASS")
print("👉 SWIPE RIGHT: Person returns to NORMAL")

# ----------------- STATE & GESTURE TRACKING -----------------
is_glass_mode = False
bg_frame = None
bg_accum = None
calibration_frames = 25
calibrated = False

# Swipe velocity tracking queue
hand_history = deque(maxlen=15)
last_swipe_time = 0.0

# Sweep transition animation
sweep_active = False
sweep_progress = 1.0
sweep_direction = "LEFT"

# Particle Sparks on Hand Trail
trail_particles = []

class TrailParticle:
    def __init__(self, x, y, color=(0, 255, 255)):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.color = color
        self.life = 1.0
        self.size = random.uniform(3, 6)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.life -= 0.06
        self.size = max(0.5, self.size * 0.94)
        return self.life > 0

    def draw(self, img):
        if self.life <= 0: return
        alpha = self.life
        col = (int(self.color[0] * alpha), int(self.color[1] * alpha), int(self.color[2] * alpha))
        cv2.circle(img, (int(self.x), int(self.y)), max(1, int(self.size)), col, -1)


# ----------------- LIQUID GLASS & OPTICAL REFRACTION RENDERER -----------------
def render_glass_person(live_img, bg_img, mask_person, t):
    """Turns the detected person into a breathtaking frosted liquid crystal glass figure
    with optical refraction displacement, chromatic prism dispersion, and glossy specular rim reflections."""
    h, w = live_img.shape[:2]
    
    # 1. Person Normal Map from Smooth Mask Gradients
    mask_blur = cv2.GaussianBlur(mask_person.astype(np.float32), (25, 25), 0)
    sobel_x = cv2.Sobel(mask_blur, cv2.CV_32F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(mask_blur, cv2.CV_32F, 0, 1, ksize=5)
    
    # 2. Refraction Mesh Displacement Grids
    grid_y, grid_x = np.indices((h, w), dtype=np.float32)
    
    # Caustic liquid wave ripples
    wave_x = np.sin(grid_y * 0.04 + t * 3.5) * 4.0
    wave_y = np.cos(grid_x * 0.04 + t * 3.5) * 4.0
    
    # Base refraction offset
    refract_power = 22.0
    dx = sobel_x * refract_power + wave_x
    dy = sobel_y * refract_power + wave_y
    
    # 3. Chromatic Dispersion (Prism Color Separation)
    dispersion = 4.0
    # Red Channel Map (Shifted Right)
    map_rx = np.clip(grid_x + dx + dispersion, 0, w - 1).astype(np.float32)
    map_ry = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)
    # Green Channel Map (Center)
    map_gx = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
    map_gy = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)
    # Blue Channel Map (Shifted Left)
    map_bx = np.clip(grid_x + dx - dispersion, 0, w - 1).astype(np.float32)
    map_by = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)
    
    source_bg = bg_img if bg_img is not None else live_img
    
    r_channel = cv2.remap(source_bg[:, :, 2], map_rx, map_ry, cv2.INTER_LINEAR)
    g_channel = cv2.remap(source_bg[:, :, 1], map_gx, map_gy, cv2.INTER_LINEAR)
    b_channel = cv2.remap(source_bg[:, :, 0], map_bx, map_by, cv2.INTER_LINEAR)
    refracted_glass = cv2.merge([b_channel, g_channel, r_channel])
    
    # 4. Frosted Glass Diffusion
    frosted = cv2.GaussianBlur(source_bg, (15, 15), 0)
    glass_body = cv2.addWeighted(refracted_glass, 0.75, frosted, 0.25, 0)
    
    # 5. Specular Gloss & Fresnel Rim Highlights
    edges = cv2.Canny(cv2.GaussianBlur(mask_person, (5, 5), 0), 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    
    # Internal body contour highlights (Facial structure / clothes edges in glass)
    gray_live = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
    body_features = cv2.Canny(cv2.GaussianBlur(gray_live, (3, 3), 0), 40, 100)
    body_features = np.where(mask_person > 0, body_features, 0)
    
    # Add glossy ice-blue / white specular reflections
    gloss_overlay = np.zeros_like(live_img)
    gloss_overlay[edges_dilated > 0] = [255, 255, 255] # Outer crystal edge
    gloss_overlay[body_features > 0] = [255, 240, 200]  # Subtle internal glass refractions
    gloss_overlay = cv2.GaussianBlur(gloss_overlay, (3, 3), 0)
    
    # Composite Final Glass Person
    glass_person = cv2.add(glass_body, gloss_overlay)
    
    # Tint subtle ice-cyan glass tone
    glass_tint = np.zeros_like(live_img)
    glass_tint[:, :] = [40, 25, 5] # Blue/cyan tint
    glass_person = cv2.add(glass_person, glass_tint)
    
    # Mask composite: Place glass person over live background
    mask_3c = (mask_person[:, :, None] > 0)
    return np.where(mask_3c, glass_person, live_img)


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    t = time.time()

    # ----------------- 1. BACKGROUND CALIBRATION -----------------
    if not calibrated:
        if bg_frame is None:
            bg_frame = img.copy()
            bg_accum = img.astype(np.float32)
        else:
            cv2.accumulateWeighted(img.astype(np.float32), bg_accum, 0.2)
            bg_frame = cv2.convertScaleAbs(bg_accum)
        
        calibration_frames -= 1
        if calibration_frames <= 0:
            calibrated = True
            print("✅ Background siap!")

    timestamp_ms = time.time_ns() // 1_000_000
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    results = landmarker.detect_for_video(mp_image, timestamp_ms)

    mask_person = None
    if segmenter is not None:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    # Adaptive background updates
    if calibrated and mask_person is not None and bg_accum is not None:
        bg_mask = (mask_person == 0).astype(np.uint8)
        if np.any(bg_mask):
            current_float = img.astype(np.float32)
            bg_accum = np.where(bg_mask[:, :, None] == 1, bg_accum * 0.98 + current_float * 0.02, bg_accum)
            bg_frame = cv2.convertScaleAbs(bg_accum)

    # ----------------- 2. HAND SWIPE VELOCITY DETECTION -----------------
    hand_centroids = []
    if results.hand_landmarks:
        for hand_lms in results.hand_landmarks:
            cx = int((hand_lms[0].x + hand_lms[5].x + hand_lms[17].x) / 3.0 * w)
            cy = int((hand_lms[0].y + hand_lms[5].y + hand_lms[17].y) / 3.0 * h)
            hand_centroids.append((cx, cy))

            # Add glowing spark particles to hand trail
            if random.random() < 0.6:
                trail_particles.append(TrailParticle(cx, cy, (255, 255, 0) if is_glass_mode else (0, 255, 128)))

        if hand_centroids:
            avg_x = sum([p[0] for p in hand_centroids]) / len(hand_centroids)
            avg_y = sum([p[1] for p in hand_centroids]) / len(hand_centroids)
            hand_history.append((avg_x, avg_y, t))

            # Detect rapid horizontal swipe over recent frames
            if len(hand_history) >= 4 and (t - last_swipe_time > 0.6):
                oldest_x, _, oldest_t = hand_history[0]
                newest_x, _, newest_t = hand_history[-1]
                dt = max(0.01, newest_t - oldest_t)
                dx = newest_x - oldest_x
                vx = dx / dt

                # 👈 SWIPE LEFT -> TURN INTO LIQUID GLASS!
                if (dx < -95 or vx < -380) and not is_glass_mode:
                    is_glass_mode = True
                    last_swipe_time = t
                    sweep_active = True
                    sweep_progress = 0.0
                    sweep_direction = "LEFT"
                    hand_history.clear()
                    print("💎 SWIPE LEFT DETECTED! Transformed into CRYSTAL GLASS!")

                # 👉 SWIPE RIGHT -> RETURN TO NORMAL!
                elif (dx > 95 or vx > 380) and is_glass_mode:
                    is_glass_mode = False
                    last_swipe_time = t
                    sweep_active = True
                    sweep_progress = 0.0
                    sweep_direction = "RIGHT"
                    hand_history.clear()
                    print("👤 SWIPE RIGHT DETECTED! Restored to NORMAL!")

    # ----------------- 3. RENDER GLASS OR NORMAL VIEW WITH SWEEP WAVE -----------------
    final_render = img.copy()

    if mask_person is not None:
        glass_frame = render_glass_person(img, bg_frame, mask_person, t)

        if is_glass_mode:
            if sweep_active:
                sweep_progress += 0.09
                if sweep_progress >= 1.0:
                    sweep_active = False
                    sweep_progress = 1.0

                sweep_x = int(w * (1.0 - sweep_progress))
                final_render[:, sweep_x:w] = glass_frame[:, sweep_x:w]
                final_render[:, 0:sweep_x] = img[:, 0:sweep_x]

                # Ice-Cyan Glass Laser Sweep Line
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (255, 255, 200), 3)
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (0, 255, 255), 1)
                for _ in range(6):
                    ry = random.randint(0, h)
                    cv2.circle(final_render, (sweep_x + random.randint(-8, 8), ry), random.randint(2, 5), (255, 255, 255), -1)
            else:
                final_render = glass_frame
        else:
            if sweep_active:
                sweep_progress += 0.09
                if sweep_progress >= 1.0:
                    sweep_active = False
                    sweep_progress = 1.0

                sweep_x = int(w * sweep_progress)
                final_render[:, 0:sweep_x] = img[:, 0:sweep_x]
                final_render[:, sweep_x:w] = glass_frame[:, sweep_x:w]

                # Emerald De-Cloak Laser Line
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (0, 255, 128), 3)
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (255, 255, 255), 1)
                for _ in range(6):
                    ry = random.randint(0, h)
                    cv2.circle(final_render, (sweep_x + random.randint(-8, 8), ry), random.randint(2, 5), (0, 255, 128), -1)
            else:
                final_render = img

    # Update & Draw Trail Particles
    trail_particles = [p for p in trail_particles if p.update() and (p.draw(final_render) or True)]

    # Draw Hand Glowing Reticles
    for cx, cy in hand_centroids:
        col = (255, 255, 0) if is_glass_mode else (0, 255, 128)
        cv2.circle(final_render, (cx, cy), 12, col, 2)
        cv2.circle(final_render, (cx, cy), 4, (255, 255, 255), -1)

    # ----------------- 4. STATUS HUD & TELEMETRY -----------------
    overlay = final_render.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (16, 16, 22), -1)
    cv2.addWeighted(overlay, 0.80, final_render, 0.20, 0, final_render)

    if is_glass_mode:
        status_badge = "💎 STATUS: LIQUID CRYSTAL GLASS [👈 SWIPED LEFT]"
        badge_color = (255, 255, 0)
    else:
        status_badge = "👤 STATUS: NORMAL HUMAN [👉 SWIPED RIGHT]"
        badge_color = (0, 255, 128)

    cv2.line(final_render, (0, 52), (w, 52), badge_color, 2)
    cv2.putText(final_render, status_badge, (16, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, badge_color, 1)

    guide_text = "👈 Swipe hand LEFT = LIQUID GLASS  |  👉 Swipe hand RIGHT = NORMAL  |  'b' = Calibrate  |  'q' = Exit"
    cv2.putText(final_render, guide_text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    cv2.imshow('RETROLENS Swipe Liquid Glass Filter', final_render)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        calibrated = False
        calibration_frames = 20
        print("🔄 Mengkalibrasi ulang background...")
    elif key == ord('g'):
        is_glass_mode = not is_glass_mode

cap.release()
cv2.destroyAllWindows()