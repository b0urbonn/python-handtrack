import os
import cv2
import mediapipe as mp
import time
import math
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
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
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

# Generate Galaxy Background
galaxy_bg = np.zeros((1080, 1920, 3), dtype=np.uint8) 
galaxy_bg[:] = (30, 10, 40) # Space purple
for _ in range(800):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    galaxy_bg[sy, sx] = (255, 255, 255)
for _ in range(100):
    sx = np.random.randint(0, 1920)
    sy = np.random.randint(0, 1080)
    cv2.circle(galaxy_bg, (sx, sy), np.random.randint(2, 6), (np.random.randint(150, 255), np.random.randint(100, 255), 255), -1)

print("Membuka Kamera... Tekan tombol 'q' di keyboard untuk keluar.")

filters = ["MONO", "DUAL-TONE", "PIXELATE", "INVERT", "SEPIA", "BLUR", "THERMAL", "SKETCH", "GLITCH", "NEON", "GALAXY"]
current_filter = 0
gesture_triggered = False

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
        filtered[mask_c == 255] = [0, 165, 255] # Orange
        filtered[mask_c == 0] = [147, 20, 255]  # Pink
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (w_r//10, h_r//10), interpolation=cv2.INTER_LINEAR)
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
        kernel = np.ones((3,3), np.uint8)
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

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1) 
    h, w, c = img.shape
    
    # Crop galaxy bg to match frame size dynamically
    frame_galaxy = galaxy_bg[:h, :w]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    timestamp_ms = time.time_ns() // 1_000_000
    
    results = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    filter_name = filters[current_filter]
    
    mask_person = None
    if filter_name == "GALAXY" and segmenter is not None:
        seg_result = segmenter.segment_for_video(mp_image, timestamp_ms)
        if seg_result.category_mask is not None:
            mask_person = seg_result.category_mask.numpy_view()
            if mask_person.shape != (h, w):
                mask_person = cv2.resize(mask_person, (w, h), interpolation=cv2.INTER_NEAREST)

    pts_portal = []
    change_filter = False

    if results.hand_landmarks:
        # Cek gestur untuk ganti filter
        if len(results.hand_landmarks) >= 2:
            idx0 = results.hand_landmarks[0][8]
            idx1 = results.hand_landmarks[1][8]
            pt0 = (int(idx0.x * w), int(idx0.y * h))
            pt1 = (int(idx1.x * w), int(idx1.y * h))
            if math.hypot(pt0[0]-pt1[0], pt0[1]-pt1[1]) < 40:
                change_filter = True
                
        # Cek gestur cubit jempol & telunjuk untuk GAMBAR
        for hand_lms in results.hand_landmarks:
            thumb = hand_lms[4]
            index = hand_lms[8]
            tx, ty = int(thumb.x * w), int(thumb.y * h)
            ix, iy = int(index.x * w), int(index.y * h)
                
            # Gestur ganti filter cadangan (Jempol & Kelingking)
            pinky = hand_lms[20]
            px, py = int(pinky.x * w), int(pinky.y * h)
            if math.hypot(tx - px, ty - py) < 40:
                change_filter = True
                
        if change_filter:
            if not gesture_triggered:
                current_filter = (current_filter + 1) % len(filters)
                gesture_triggered = True
        else:
            gesture_triggered = False

        for hand_lms in results.hand_landmarks:
            for id, lm in enumerate(hand_lms):
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [4, 8]:
                    pts_portal.append([cx, cy])
                    cv2.circle(img, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

        # 4 POIN -> PORTAL MIRING (Polygon)
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
            
            if bw > 0 and bh > 0:
                roi = img[y:y+bh, x:x+bw].copy()
                filtered_roi = apply_filter(roi, filter_name, x, y, mask_person, frame_galaxy)
                
                mask = np.zeros((bh, bw), dtype=np.uint8)
                poly_roi = poly_pts - [x, y]
                cv2.fillPoly(mask, [poly_roi], 255)
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                img[y:y+bh, x:x+bw] = np.where(mask_3ch == 255, filtered_roi, roi)
                
                cv2.polylines(img, [poly_pts], True, (255, 255, 255), 2)
                
                # --- PARTIKEL GLOW DI TEPI PORTAL ---
                for i in range(4):
                    pt1 = poly_pts[i]
                    pt2 = poly_pts[(i+1)%4]
                    for _ in range(5):
                        alpha = np.random.random()
                        px = int(pt1[0] * alpha + pt2[0] * (1 - alpha)) + np.random.randint(-15, 15)
                        py = int(pt1[1] * alpha + pt2[1] * (1 - alpha)) + np.random.randint(-15, 15)
                        cv2.circle(img, (px, py), np.random.randint(1, 4), (0, 255, 255), -1)

                cv2.putText(img, f"PORTAL: {filter_name}", (top_pts[0][0], top_pts[0][1] - 10), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

    # Overlay Teks
    cv2.putText(img, "Ganti Filter: Jempol & Kelingking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Filter Aktif: {filters[current_filter]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('RETROLENS Pake Python', img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()