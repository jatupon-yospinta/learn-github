import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model (~2 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

WRIST = 0; THUMB_TIP = 4; THUMB_IP = 3
INDEX_TIP = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

PINCH_THRESHOLD = 45

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


# ─── Target Box ────────────────────────────────────────────────────────────────

class TargetBox:
    def __init__(self, x, y, w, h, label=""):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label       = label
        self.occupied_by = None   # VirtualObject placed inside

    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, px, py):
        return self.x <= px <= self.x + self.w and self.y <= py <= self.y + self.h

    def draw(self, frame, hover=False):
        x, y, w, h = self.x, self.y, self.w, self.h
        filled = self.occupied_by is not None

        # Background fill
        overlay = frame.copy()
        bg_color = (0, 60, 0) if filled else ((0, 40, 60) if hover else (30, 30, 30))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), bg_color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Border
        border = (0, 230, 100) if filled else ((0, 180, 255) if hover else (120, 120, 120))
        thick  = 3 if (filled or hover) else 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, thick)

        # Dashed corner accents
        corner_len = 12
        for cx_, cy_ in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = 1 if cx_ == x else -1
            dy = 1 if cy_ == y else -1
            cv2.line(frame, (cx_, cy_), (cx_ + dx*corner_len, cy_), border, 2)
            cv2.line(frame, (cx_, cy_), (cx_, cy_ + dy*corner_len), border, 2)

        # Label at top of box
        if self.label:
            (tw, _), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.putText(frame, self.label, (x + w//2 - tw//2, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, border, 2)

        # Checkmark when filled
        if filled:
            cx_ = x + w // 2
            cy_ = y + h // 2
            cv2.putText(frame, "✓", (cx_ - 12, cy_ + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 100), 2)


# ─── Virtual Object ────────────────────────────────────────────────────────────

class VirtualObject:
    def __init__(self, x, y, radius, color, label=""):
        self.x = float(x)
        self.y = float(y)
        self.radius     = radius
        self.color      = color
        self.label      = label
        self.grabbed_by = None
        self.in_box     = None   # TargetBox this object is placed in

    def draw(self, frame):
        cx, cy  = int(self.x), int(self.y)
        grabbed = self.grabbed_by is not None

        # Shadow
        sx = cx + (8 if grabbed else 3)
        sy = cy + (10 if grabbed else 4)
        cv2.circle(frame, (sx, sy), self.radius + (4 if grabbed else 0), (20, 20, 20), -1)

        # Body
        cv2.circle(frame, (cx, cy), self.radius, self.color, -1)

        # Highlight
        hi = tuple(min(255, int(v * 1.5)) for v in self.color)
        cv2.circle(frame, (cx - self.radius//4, cy - self.radius//4),
                   self.radius // 3, hi, -1)

        # Border
        border = (0, 255, 255) if grabbed else (255, 255, 255)
        cv2.circle(frame, (cx, cy), self.radius, border, 3 if grabbed else 2)

        # Label
        if self.label:
            (tw, _), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(frame, self.label, (cx - tw//2, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def hit_test(self, px, py):
        return np.hypot(px - self.x, py - self.y) <= self.radius + 20


# ─── Helpers ───────────────────────────────────────────────────────────────────

def landmark_px(landmarks, idx, w, h):
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def get_pinch(landmarks, w, h):
    tx, ty = landmark_px(landmarks, THUMB_TIP, w, h)
    ix, iy = landmark_px(landmarks, INDEX_TIP,  w, h)
    dist   = np.hypot(tx - ix, ty - iy)
    return (tx+ix)//2, (ty+iy)//2, dist <= PINCH_THRESHOLD, (tx,ty), (ix,iy)


def draw_hand(frame, landmarks, w, h, color):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255,255,255), -1)
        cv2.circle(frame, pt, 4, color, 1)


def draw_pinch_indicator(frame, thumb_pt, index_pt, mid, is_pinching):
    c = (0, 255, 80) if is_pinching else (180, 180, 180)
    cv2.line(frame, thumb_pt, index_pt, c, 2)
    cv2.circle(frame, thumb_pt,  10, c, -1)
    cv2.circle(frame, index_pt, 10, c, -1)
    if is_pinching:
        cv2.circle(frame, mid, 7, (0, 255, 0), -1)


def create_objects(w, h):
    return [
        VirtualObject(w*0.12, h*0.35, 42, ( 60,  80, 220), "A"),
        VirtualObject(w*0.12, h*0.55, 38, ( 50, 180,  80), "B"),
        VirtualObject(w*0.22, h*0.42, 42, (200,  70,  60), "C"),
        VirtualObject(w*0.22, h*0.62, 38, (180, 140,  40), "D"),
    ]


def create_boxes(w, h):
    bw, bh   = 110, 90
    gap      = 30
    total    = 4 * bw + 3 * gap
    start_x  = (w - total) // 2
    y        = 50
    labels   = ["Box 1", "Box 2", "Box 3", "Box 4"]
    return [
        TargetBox(start_x + i*(bw+gap), y, bw, bh, labels[i])
        for i in range(4)
    ]


def draw_score(frame, objects, w):
    placed = sum(1 for o in objects if o.in_box is not None)
    total  = len(objects)
    text   = f"Score: {placed} / {total}"
    color  = (0, 230, 100) if placed == total else (200, 200, 200)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text, (w - tw - 15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    objects        = create_objects(W, H)
    boxes          = create_boxes(W, H)
    prev_pinching  = {}
    grabbed        = {}
    start_time     = time.time()

    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            ts_ms    = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = detector.detect_for_video(mp_image, ts_ms)

            # Collect pinch midpoints to highlight hover
            pinch_mids = []
            if result.hand_landmarks:
                for hand_lm, _ in zip(result.hand_landmarks, result.handedness):
                    mx, my, is_p, _, _ = get_pinch(hand_lm, w, h)
                    if is_p:
                        pinch_mids.append((mx, my))

            # Draw boxes (below everything)
            for box in boxes:
                hover = any(box.contains(px, py) for px, py in pinch_mids)
                box.draw(frame, hover=hover)

            # Draw objects
            for obj in objects:
                obj.draw(frame)

            active_hands = set()

            if result.hand_landmarks:
                for idx, (hand_lm, handedness) in enumerate(
                    zip(result.hand_landmarks, result.handedness)
                ):
                    label   = handedness[0].category_name
                    hand_id = f"{label}_{idx}"
                    active_hands.add(hand_id)

                    if hand_id not in prev_pinching:
                        prev_pinching[hand_id] = False
                        grabbed[hand_id]        = None

                    mx, my, is_pinching, thumb_pt, index_pt = get_pinch(hand_lm, w, h)
                    color = (0, 200, 255) if label == "Right" else (255, 100, 0)

                    draw_hand(frame, hand_lm, w, h, color)
                    draw_pinch_indicator(frame, thumb_pt, index_pt, (mx, my), is_pinching)

                    # Pinch START → grab object (free it from box if needed)
                    if is_pinching and not prev_pinching[hand_id]:
                        for obj in objects:
                            if obj.grabbed_by is None and obj.hit_test(mx, my):
                                grabbed[hand_id] = obj
                                obj.grabbed_by   = hand_id
                                if obj.in_box:
                                    obj.in_box.occupied_by = None
                                    obj.in_box             = None
                                break

                    # Holding → drag
                    if is_pinching and grabbed[hand_id]:
                        grabbed[hand_id].x = float(mx)
                        grabbed[hand_id].y = float(my)

                    # Pinch END → try to drop into a box
                    if not is_pinching and prev_pinching[hand_id]:
                        obj = grabbed[hand_id]
                        if obj:
                            placed = False
                            for box in boxes:
                                if box.occupied_by is None and box.contains(int(obj.x), int(obj.y)):
                                    # Snap to box center
                                    cx_, cy_       = box.center()
                                    obj.x          = float(cx_)
                                    obj.y          = float(cy_)
                                    box.occupied_by = obj
                                    obj.in_box      = box
                                    placed = True
                                    break
                            obj.grabbed_by   = None
                            grabbed[hand_id] = None

                    prev_pinching[hand_id] = is_pinching

                    status = "GRABBING" if grabbed[hand_id] else ("Pinch" if is_pinching else "Open")
                    tx = 12 if label == "Left" else w - 200
                    cv2.putText(frame, f"{label}: {status}",
                                (tx, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Release objects from vanished hands
            for hand_id in list(prev_pinching):
                if hand_id not in active_hands:
                    if grabbed.get(hand_id):
                        grabbed[hand_id].grabbed_by = None
                    grabbed.pop(hand_id, None)
                    prev_pinching.pop(hand_id, None)

            # Header bar
            bar = frame.copy()
            cv2.rectangle(bar, (0, 0), (w, 40), (30, 30, 30), -1)
            cv2.addWeighted(bar, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, "Pinch to grab  |  Drop into box  |  Q to quit",
                        (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            draw_score(frame, objects, w)

            cv2.imshow("Hand Motion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
