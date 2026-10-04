"""
Pre-extract 64 head-rotation WebP frames (+ center.webp) from the character video.
Run once:  python3 extract_frames.py <video.mp4>
Nothing is ever seeked/played in the browser - the page only draws still WebP frames.
"""
import sys, os, cv2, numpy as np

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "character.mp4"
OUT = "frame"
N = 64                       # frames around the 360 degree circle (5.625 deg apart)
CROP_X0, CROP_X1 = 480, 1440  # centre 960px of the 1920px frame (the character)
OUT_W, OUT_H = 800, 900
CENTER_FRAME = 239           # neutral pose looking into the camera (last frame)

# Timeline found by inspecting the clip: (screen angle in degrees, video frame)
# screen angle: 0 = RIGHT, 90 = DOWN, 180 = LEFT, 270/-90 = UP  (atan2(dy, dx), y down)
KEYS = [
    (-90, 30),    # UP
    (-45, 50),    # UP-RIGHT
    (0,   75),    # RIGHT
    (45,  100),   # DOWN-RIGHT
    (90,  125),   # DOWN
    (135, 150),   # DOWN-LEFT
    (180, 175),   # LEFT
    (225, 195),   # UP-LEFT
]

def frame_for_angle(a):
    a = (a + 90) % 360 - 90            # normalise to [-90, 270)
    for (a0, f0), (a1, f1) in zip(KEYS, KEYS[1:]):
        if a0 <= a <= a1:
            return int(round(f0 + (f1 - f0) * (a - a0) / (a1 - a0)))
    # gap between UP-LEFT (225) and UP (270): the clip never shows it, so snap to nearest pose
    return KEYS[-1][1] if a < 247.5 else KEYS[0][1]

angles = [-90 + i * 360 / N for i in range(N)]
plan = {i: frame_for_angle(a) for i, a in enumerate(angles)}
needed = set(plan.values()) | {CENTER_FRAME}

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frames in video:", total, "fps:", cap.get(cv2.CAP_PROP_FPS))

grabbed, border = {}, []
for idx in range(total):
    ok, frame = cap.read()
    if not ok: break
    crop = frame[:, CROP_X0:CROP_X1]
    # sample the plain backdrop (top strip + both sides) to detect the exact background colour
    # backdrop sample: top strip + upper half of both sides (below that is the sweater)
    hh = crop.shape[0] // 2
    border.append(np.concatenate([crop[:16].reshape(-1, 3), crop[:hh, :16].reshape(-1, 3), crop[:hh, -16:].reshape(-1, 3)]).mean(0))
    if idx in needed:
        grabbed[idx] = crop

b, g, r = np.median(np.array(border), axis=0)
print("BACKGROUND HEX: #%02X%02X%02X" % (round(r), round(g), round(b)))

os.makedirs(OUT, exist_ok=True)
def save(img, path):
    img = cv2.resize(img, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_WEBP_QUALITY, 88])

for i, f in plan.items():
    save(grabbed[f], f"{OUT}/f{i:02d}.webp")
save(grabbed[CENTER_FRAME], f"{OUT}/center.webp")
print("wrote", N, "frames + center.webp")
