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
    print("Segmenter tidak tersedia (pastikan file selfie_segmenter.tflite ada)", e)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera.")
    exit()

print("✨ GESTURE SWIPE INVISIBILITY CLOAK STUDIO ACTIVE ✨")
print("👈 SWIPE LEFT: Person becomes INVISIBLE")
print("👉 SWIPE RIGHT: Person returns to NORMAL")

# ----------------- STATE & GESTURE TRACKING -----------------
is_invisible = False
bg_frame = None
bg_accum = None
calibration_frames = 25
calibrated = False

# Swipe velocity tracking queue (stores (x, y, timestamp))
hand_history = deque(maxlen=15)
last_swipe_time = 0.0

# Wipe animation state
sweep_active = False
sweep_progress = 1.0
sweep_direction = "LEFT" # "LEFT" (to invisible) or "RIGHT" (to visible)

# Particle Sparks along Hand Trail
trail_particles = []

class TrailParticle:
    def __init__(self, x, y, color=(0, 255, 255)):
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-2, 2)
        self.vy = random.uniform(-2, 2)
        self.color = color
        self.life = 1.0
        self.size = random.uniform(3, 7)

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

    # Adaptive background updates on non-person areas
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
            # Palm center (average of wrist 0, index MCP 5, pinky MCP 17)
            cx = int((hand_lms[0].x + hand_lms[5].x + hand_lms[17].x) / 3.0 * w)
            cy = int((hand_lms[0].y + hand_lms[5].y + hand_lms[17].y) / 3.0 * h)
            hand_centroids.append((cx, cy))

            # Add glowing spark particles to hand trail
            if random.random() < 0.6:
                trail_particles.append(TrailParticle(cx, cy, (0, 255, 255) if is_invisible else (0, 255, 128)))

        if hand_centroids:
            # Average X across visible hands
            avg_x = sum([p[0] for p in hand_centroids]) / len(hand_centroids)
            avg_y = sum([p[1] for p in hand_centroids]) / len(hand_centroids)
            hand_history.append((avg_x, avg_y, t))

            # Detect rapid horizontal swipe over recent frames
            if len(hand_history) >= 4 and (t - last_swipe_time > 0.6):
                oldest_x, _, oldest_t = hand_history[0]
                newest_x, _, newest_t = hand_history[-1]
                dt = max(0.01, newest_t - oldest_t)
                dx = newest_x - oldest_x
                vx = dx / dt # Pixels per second

                # 👈 SWIPE LEFT (dx < -100px or vx < -350px/s) -> GO INVISIBLE!
                if (dx < -95 or vx < -380) and not is_invisible:
                    is_invisible = True
                    last_swipe_time = t
                    sweep_active = True
                    sweep_progress = 0.0
                    sweep_direction = "LEFT"
                    hand_history.clear()
                    print("🫥 SWIPE LEFT DETECTED! Cloaking Activated -> INVISIBLE!")

                # 👉 SWIPE RIGHT (dx > 100px or vx > 350px/s) -> GO NORMAL / VISIBLE!
                elif (dx > 95 or vx > 380) and is_invisible:
                    is_invisible = False
                    last_swipe_time = t
                    sweep_active = True
                    sweep_progress = 0.0
                    sweep_direction = "RIGHT"
                    hand_history.clear()
                    print("👤 SWIPE RIGHT DETECTED! Cloaking Deactivated -> NORMAL / VISIBLE!")

    # ----------------- 3. INVISIBILITY CLOAK RENDERING -----------------
    final_render = img.copy()

    if bg_frame is not None and mask_person is not None:
        # Full invisible frame (person silhouette replaced by real background)
        invisible_frame = np.where(mask_person[:, :, None] > 0, bg_frame, img)

        if is_invisible:
            if sweep_active:
                # Animated Cloaking Sweep from Right to Left
                sweep_progress += 0.09
                if sweep_progress >= 1.0:
                    sweep_active = False
                    sweep_progress = 1.0

                sweep_x = int(w * (1.0 - sweep_progress))
                # Left of sweep line is already cloaked
                final_render[:, sweep_x:w] = invisible_frame[:, sweep_x:w]
                final_render[:, 0:sweep_x] = img[:, 0:sweep_x]

                # Electric Cyan Sweep Laser Line
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (0, 255, 255), 3)
                cv2.line(final_render, (sweep_x, 0), (sweep_x, h), (255, 255, 255), 1)
                for _ in range(6):
                    ry = random.randint(0, h)
                    cv2.circle(final_render, (sweep_x + random.randint(-8, 8), ry), random.randint(2, 5), (0, 255, 255), -1)
            else:
                final_render = invisible_frame
        else:
            if sweep_active:
                # Animated De-Cloaking Sweep from Left to Right
                sweep_progress += 0.09
                if sweep_progress >= 1.0:
                    sweep_active = False
                    sweep_progress = 1.0

                sweep_x = int(w * sweep_progress)
                # Left of sweep line is restored to normal
                final_render[:, 0:sweep_x] = img[:, 0:sweep_x]
                final_render[:, sweep_x:w] = invisible_frame[:, sweep_x:w]

                # Electric Emerald De-Cloak Laser Line
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
        color = (0, 255, 255) if is_invisible else (0, 255, 128)
        cv2.circle(final_render, (cx, cy), 12, color, 2)
        cv2.circle(final_render, (cx, cy), 4, (255, 255, 255), -1)

    # ----------------- 4. SLEEK STATUS HUD & GESTURE TELEMETRY -----------------
    overlay = final_render.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (16, 16, 22), -1)
    cv2.addWeighted(overlay, 0.80, final_render, 0.20, 0, final_render)

    if is_invisible:
        status_badge = "🫥 STATUS: INVISIBLE (CLOAKED) [👈 SWIPED LEFT]"
        badge_color = (0, 255, 255)
    else:
        status_badge = "👤 STATUS: NORMAL (VISIBLE) [👉 SWIPED RIGHT]"
        badge_color = (0, 255, 128)

    cv2.line(final_render, (0, 52), (w, 52), badge_color, 2)
    cv2.putText(final_render, status_badge, (16, 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, badge_color, 1)

    guide_text = "👈 Swipe hand LEFT = INVISIBLE  |  👉 Swipe hand RIGHT = NORMAL  |  'b' = Calibrate BG  |  'q' = Exit"
    cv2.putText(final_render, guide_text, (16, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1)

    cv2.imshow('RETROLENS Swipe Invisibility Cloak', final_render)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        calibrated = False
        calibration_frames = 20
        print("🔄 Mengkalibrasi ulang background...")
    elif key == ord('i'):
        # Keyboard fallback toggle
        is_invisible = not is_invisible

cap.release()
cv2.destroyAllWindows()