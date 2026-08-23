import os
import cv2
import mediapipe as mp
import time
import math
import random
import numpy as np

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

print("✨ INVISIBLE PORTAL STUDIO (OPTICAL CAMOUFLAGE / INVISIBILITY CLOAK) ACTIVE ✨")

bg_frame = None
bg_accum = None
calibration_frames = 25
calibrated = False
last_recalib_time = 0.0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    t = time.time()

    # ----------------- 1. BACKGROUND CAPTURE & ADAPTIVE ACCUMULATION -----------------
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
            print("✅ Background berhasil dikalibrasi!")

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

    # Adaptively update background where no person is detected
    if calibrated and mask_person is not None and bg_accum is not None:
        bg_mask = (mask_person == 0).astype(np.uint8)
        if np.any(bg_mask):
            current_float = img.astype(np.float32)
            bg_accum = np.where(bg_mask[:, :, None] == 1, bg_accum * 0.98 + current_float * 0.02, bg_accum)
            bg_frame = cv2.convertScaleAbs(bg_accum)

    pts_portal = []
    recalibrate_trigger = False

    if results.hand_landmarks:
        # Cek gestur re-kalibrasi background: Sentuh kedua jempol (< 35px)
        if len(results.hand_landmarks) >= 2:
            t0 = results.hand_landmarks[0][4]
            t1 = results.hand_landmarks[1][4]
            pt0 = (int(t0.x * w), int(t0.y * h))
            pt1 = (int(t1.x * w), int(t1.y * h))
            if math.hypot(pt0[0] - pt1[0], pt0[1] - pt1[1]) < 35:
                if t - last_recalib_time > 1.5:
                    calibrated = False
                    calibration_frames = 20
                    last_recalib_time = t
                    print("🔄 Mengkalibrasi ulang background...")

        # ----------------- ALWAYS 4 POINTS (THUMB & INDEX ON TWO HANDS) -----------------
        for hand_lms in results.hand_landmarks:
            for id_lm in [4, 8]:
                cx = int(hand_lms[id_lm].x * w)
                cy = int(hand_lms[id_lm].y * h)
                pts_portal.append([cx, cy])
                cv2.circle(img, (cx, cy), 8, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1)

        # 4 POIN -> INVISIBLE PORTAL SEGIEMPAT (DUA TANGAN)
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
            
            if bw > 0 and bh > 0 and bg_frame is not None:
                roi_live = img[y:y+bh, x:x+bw].copy()
                roi_bg = bg_frame[y:y+bh, x:x+bw].copy()

                # ----------------- INVISIBILITY CLOAK LOGIC -----------------
                # Render clean background inside the person silhouette to make the user completely invisible!
                if mask_person is not None:
                    roi_mask = mask_person[y:y+bh, x:x+bw]
                    # Person body becomes see-through background
                    invisible_roi = np.where(roi_mask[:, :, None] > 0, roi_bg, roi_live)
                else:
                    invisible_roi = roi_bg

                # Subtle Optical Refraction Shimmer on Cloaking Field
                shimmer_dx = (np.sin(np.linspace(0, 10, bw)[None, :] + t * 4.0) * 3.0).astype(np.float32)
                shimmer_dy = (np.cos(np.linspace(0, 10, bh)[:, None] + t * 4.0) * 3.0).astype(np.float32)
                grid_y, grid_x = np.indices((bh, bw), dtype=np.float32)
                map_x = np.clip(grid_x + shimmer_dx, 0, bw - 1).astype(np.float32)
                map_y = np.clip(grid_y + shimmer_dy, 0, bh - 1).astype(np.float32)
                shimmered_invisible = cv2.remap(invisible_roi, map_x, map_y, cv2.INTER_LINEAR)
                invisible_roi = cv2.addWeighted(invisible_roi, 0.85, shimmered_invisible, 0.15, 0)

                # Composite inside the 4-point polygon portal
                mask_poly = np.zeros((bh, bw), dtype=np.uint8)
                poly_roi = poly_pts - [x, y]
                cv2.fillPoly(mask_poly, [poly_roi], 255)
                mask_3ch = cv2.cvtColor(mask_poly, cv2.COLOR_GRAY2BGR)
                
                img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, invisible_roi, roi_live)
                
                # --- GLOWING CAMOUFLAGE PORTAL RIM ---
                cv2.polylines(img, [poly_pts], True, (0, 255, 255), 2)
                cv2.polylines(img, [poly_pts], True, (255, 255, 255), 1)
                
                # Cloaking Energy Sparks along Perimeter
                for i in range(4):
                    pt1 = poly_pts[i]
                    pt2 = poly_pts[(i+1)%4]
                    for _ in range(4):
                        alpha = np.random.random()
                        px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-10, 10)
                        py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-10, 10)
                        cv2.circle(img, (px, py), np.random.randint(1, 3), (0, 255, 255), -1)

                cv2.putText(img, "PORTAL: INVISIBLE CLOAK 🫥", (top_pts[0][0], max(30, top_pts[0][1] - 10)), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)

    # ----------------- CLEAN HUD OVERLAY -----------------
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (16, 16, 22), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.line(img, (0, 42), (w, 42), (0, 255, 255), 1)

    hud_title = "🫥 INVISIBLE FILTER PORTAL  |  🖐️ Gunakan 2 Tangan (Jempol & Telunjuk)  |  'b' = Rekalibrasi Background  |  'q' = Keluar"
    cv2.putText(img, hud_title, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.imshow('RETROLENS Invisible Filter Portal', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        calibrated = False
        calibration_frames = 20
        print("🔄 Mengkalibrasi ulang background...")

cap.release()
cv2.destroyAllWindows()