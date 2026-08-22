# Retrolens FX - Interactive Hand Gesture & Magic Studio ✨

An augmented reality hand-tracking studio with interactive gestures, air drawing canvas, cyber superpower visual effects, and dynamic portals powered by MediaPipe and OpenCV.

---

## 🚀 Features & Modes

### 1. ⚽ INTERACTIVE PHYSICS BALL ARCADE (New!)
- **Volley & Juggle with Hands**: Hit, slap, and juggle the bouncing neon ball using your hands, palms, or fingertips in real-time!
- **Headbutt the Ball**: Bounce the ball off your forehead/face for bonus combo points!
- **Basketball Hoop Goal**: Aim and shoot the ball through the glowing hoop on the left for a **`SWISH! +5 PTS`** fireworks explosion!
- **Scoreboard & Combos**: Live combo streak multiplier (`JUGGLES`, `ON FIRE!`, `UNSTOPPABLE!`) and high score tracking.
- **Ball Types**: Switch between **`NEON`**, **`FIREBALL`** (blazing flame trail), and **`PLASMA`**!
- **Reset Button**: Touchless hover/pinch on **`RESET ⚽`** to drop a new ball anytime.

### 2. 🤖 AR FACE TRACKING & 478-POINT FULL FACE DOTS
- 🌟 **FACE DOTS (Default)**: Full **478 3D facial landmark mocap dots** tracking your entire face in real-time with color-coded features:
  - 🟣 **Lips & Mouth**: Neon Magenta dots + lip contours
  - 🔵 **Eyes & Pupils**: Neon Cyan dots + precision tracking
  - 🟡 **Eyebrows**: Golden Yellow dots
  - 🟠 **Nose**: High-tech Orange tracking nodes
  - 🟢 **Facial Oval & Cheeks**: Cyber Matrix Green mocap points
- 🌐 **CYBER MESH**: Futuristic holographic 3D wireframe mesh connected over face contours!
- 🕶️ **CYBER VISOR**: Glowing futuristic HUD glasses with live telemetry, scanlines, and target locks!
- 🥽 **IRON MAN HUD**: Tactical facial tracking brackets, eye reticles, and diagnostics!
- 👑 **NEON CROWN & HALO**: Floating glowing golden halo with orbiting sparkle stars over your head!
- 🐱 **CYBER CAT EARS**: Holographic neon ears & whiskers that track your face in real-time!
- 🔥 **LASER EYES**: Blazing plasma laser beams shooting forward from your pupils!
- 🎭 **Touchless Switcher**: Hover/pinch the **`FACE: ...`** button on top to cycle face filters!

### 2. 🎨 IN-CAMERA AIR DRAWING (New Brushes & Smoothing!)
- 👆 **Index Finger Up**: Draw glowing neon lines anywhere directly on the live camera feed!
- 🔥 **Fire & Ember Brush**: Spawns living animated fire particles along your drawn strokes!
- 🌈 **Rainbow Mode**: Dynamic chromatic shifting colors!
- ✌️ **Peace / V-Sign**: Hover navigation reticle (pointer without drawing).
- 🖐️ **Open Palm**: Physical eraser that wipes drawings clean under your palm.
- 🎨 **Top Color Palette**: Hover or pinch over any color button (`CYA`, `PIN`, `GRE`, `YEL`, `ORA`, `RED`, `RAI`, `FIR`, `ERA`).
- ↩️ **Touchless Undo & Clear**: Instant touchless buttons for Undo and Clear.

### 3. ⚡ MAGIC FX Mode (Cyber Spells & Superpowers)
- ⚡ **Finger Gun** (Thumb & Index extended): Charges and shoots animated **Laser Beams** with particles!
- 🤘 **Rock-On / Metal Sign** (Index & Pinky up): Generates crackling **Electric Lightning Arcs** between fingertips!
- 💥 **Kamehameha / Plasma Energy Orb**: Bring **both hands** close together facing each other to create a pulsing plasma energy sphere!
- 👍 **Thumbs Up**: Triggers colorful **Fireworks & Confetti** explosions!
- 🖐️ **Open Palm**: Casts a holographic **Rotating Runic Magic Shield**!

### 4. 🔮 RETRO PORTAL Mode (Camera Filter Warps)
- **Open Portal**: Bring thumb & index of **both hands** together (4 finger points) to warp space with 11 custom camera filters:
  - `MONO`, `DUAL-TONE`, `PIXELATE`, `INVERT`, `SEPIA`, `BLUR`, `THERMAL`, `SKETCH`, `GLITCH`, `NEON`, `GALAXY`
- **Switch Filter**: Touch your thumb and pinky finger together or click **`◀ PREV`** / **`NEXT ▶`**.

---

## 🎮 Keyboard Controls
| Key | Action |
| --- | --- |
| **`M`** | Switch Active Mode (`Magic FX` ↔ `Air Canvas` ↔ `Retro Portal`) |
| **`1`, `2`, `3`** | Jump directly to Mode 1, 2, or 3 |
| **`S`** | 📸 Take **High-Res Screenshot** (saved to `screenshots/` with camera shutter flash) |
| **`C`** | 🧹 Clear Canvas (with confetti burst) |
| **`U`** | ↩️ Undo Last Drawing Stroke |
| **`Q`** | 🚪 Exit Application |

---

## 🏃 How to Run

```bash
cd /Users/raphaeldaleogbac/.gemini/antigravity-ide/scratch/python-handtrack
./run.sh
```

Or directly via python:
```bash
/Users/raphaeldaleogbac/.gemini/antigravity-ide/scratch/python-handtrack/.venv/bin/python main.py
```