"""Hand_Tester.py - live-match your hand against signs.jsonl and show the result on the HUD

Keys (click the camera window first):
    A-Z / 0-9   set the sign you EXPECT to be making (HUD turns green/red and tracks hit rate)
    SPACE       clear the expected sign
    ESC         quit
"""

import json
import time
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
SAMPLES_FILE = "signs.jsonl"
K = 5
UNKNOWN_DIST = 1.0    # nearest sample farther than this -> "?"; tune using the on-screen dist
SMOOTH_FRAMES = 7     # show the most common result of the last N frames (stops flicker)


def create_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize(landmarks):
    # MUST be identical to the recorder's normalize
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        return None
    pts /= scale
    return pts.flatten()


def load_samples(path):
    X, labels, hands = [], [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                X.append(s["landmarks"])
                labels.append(s["label"])
                hands.append(s["hand"])
    return np.array(X), np.array(labels), np.array(hands)


def predict(vec, hand, X, labels, hands):
    """Returns (label or None, nearest distance, vote confidence)."""
    mask = hands == hand                      # only compare same-handed samples
    if not mask.any():
        return None, float("inf"), 0.0
    dists = np.linalg.norm(X[mask] - vec, axis=1)
    order = np.argsort(dists)[:K]
    nearest = dists[order[0]]
    if nearest > UNKNOWN_DIST:
        return None, nearest, 0.0
    votes = Counter(labels[mask][order])
    best, count = votes.most_common(1)[0]
    return best, nearest, count / len(order)


def draw_points(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 200, 255), -1)


def draw_hud(frame, shown, dist, conf, expected, hits, total):
    h, w = frame.shape[:2]
    if expected and shown is not None:
        color = (0, 200, 0) if shown == expected else (0, 0, 255)
    else:
        color = (255, 255, 255)

    # big detected sign
    cv2.putText(frame, shown or "?", (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                4, color, 8)
    # details
    if dist is not None:
        cv2.putText(frame, f"dist={dist:.2f}  conf={conf:.0%}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # expected-sign check
    if expected:
        rate = f"{hits / total:.0%}" if total else "-"
        cv2.putText(frame, f"Expected: {expected}   hit rate: {rate}", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "Press a letter to set the expected sign", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def main():
    X, labels, hands = load_samples(SAMPLES_FILE)
    print(f"Loaded {len(X)} samples: {dict(Counter(labels))}")

    cap = cv2.VideoCapture(0)
    landmarker = create_landmarker()
    start = time.time()

    recent = deque(maxlen=SMOOTH_FRAMES)
    expected = None
    hits = total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # same flip as the recorder

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int((time.time() - start) * 1000))

        shown, dist, conf = None, None, 0.0
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            hand = result.handedness[0][0].category_name
            draw_points(frame, landmarks)

            vec = normalize(landmarks)
            if vec is not None:
                label, dist, conf = predict(vec, hand, X, labels, hands)
                recent.append(label)
                shown = Counter(recent).most_common(1)[0][0]  # smoothed result
                if expected:
                    total += 1
                    hits += (shown == expected)
        else:
            recent.clear()

        draw_hud(frame, shown, dist, conf, expected, hits, total)
        cv2.imshow("Hand Tester", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                               # ESC = quit
            break
        elif key == 32:                             # SPACE = clear expected
            expected, hits, total = None, 0, 0

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()