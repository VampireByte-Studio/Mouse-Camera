import json

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui
import numpy as np
from collections import Counter, deque


K = 5
UNKNOWN_DIST = 1.0
SMOOTH_FRAMES = 7
SAMPLES_FILE = "signs.jsonl"
#ASL related values
ASL = True
#time related values
click_cd = 0.4
type_cd = 1
mode_cd = 2
lastClickTime = 0
lastModeTime = 0
lastTypeTime = 0
#general
screen_w, screen_h = pyautogui.size()
# webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No capturing devices found")

# hands
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)
detector = vision.HandLandmarker.create_from_options(options)
#kalman filter
#turn off manual filter
pyautogui.PAUSE = 0
kf = cv2.KalmanFilter(4, 2)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)
kf.processNoiseCov = np.eye(4, dtype=np.float32)
kf.measurementNoiseCov = np.eye(2, dtype=np.float32)
kf.errorCovPost        = np.eye(4, dtype=np.float32)
kf.statePost = np.array([[screen_w // 2], [screen_h // 2], [0], [0]], np.float32)

def smooth(x, y):
    kf.predict()
    est = kf.correct(np.array([[x], [y]], np.float32))
    return int(est[0, 0]), int(est[1, 0])

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

#draw landmarks
def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape

    for i, lm in enumerate(hand_landmarks):
        cx, cy = int(lm.x * w), int(lm.y * h)

        if i == 8:
            color = (0, 255, 0)   #green - index tip
        elif i == 12:
            color = (255, 0, 0)   #blue - middle tip
        elif i == 16:
            color = (255, 255, 0)   #blue - middle tip
        elif i == 4 or i == 20:
            color = (0, 255, 255) #yellow - thumb and pinky tip
        else:
            color = (0, 0, 255)   #red - other landmarks

        cv2.circle(frame, (cx, cy), 5, color, -1)
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
#map landmarks

def map_to_screen(norm_x, norm_y, screen_w, screen_h):
    screen_x = int(norm_x * screen_w)
    screen_y = int(norm_y * screen_h)
    return screen_x, screen_y

#main loop

while True:
    success, frame = cap.read()
    if not success:
        break
    X, labels, hands = load_samples(SAMPLES_FILE)
    recent = deque(maxlen=SMOOTH_FRAMES)
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    hand = result.hand_landmarks[0]
    handedness = result.handedness[0][0].category_name
    index_tip = hand[8]

    shown = None
    vec = normalize(hand)
    if vec is not None:
        label, dist, conf = predict(vec, handedness, X, labels, hands)
        recent.append(label)
        shown = Counter(recent).most_common(1)[0][0]


    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_landmarks(frame, hand)

        h, w, _ = frame.shape
        # ... your y_diff / x_diff calcs if still needed ...

        # mouse mode
        now = time.time()

        if not ASL:
            if shown == "1" and now - lastClickTime > click_cd:
                pyautogui.leftClick()
                lastClickTime = now
                print("Left click detected")
            elif shown == "2" and now - lastClickTime > click_cd:
                pyautogui.rightClick()
                lastClickTime = now
                print("Right click detected")
            elif shown == "3" and now - lastModeTime > mode_cd:
                ASL = True
                lastModeTime = now
                print("KEYBOARD ON")

            # cursor movement belongs inside mouse mode
            screen_x, screen_y = map_to_screen(index_tip.x, index_tip.y, screen_w, screen_h)
            sx, sy = smooth(screen_x, screen_y)
            pyautogui.moveTo(sx, sy)

        else:
            if shown == "MODE" and now - lastModeTime > mode_cd:
                ASL = False
                lastModeTime = now
                print("KEYBOARD OFF")
            elif shown is not None and shown.isalpha() and len(shown) == 1 \
                    and now - lastTypeTime > type_cd:
                pyautogui.write(shown.lower())
                lastTypeTime = now



    cv2.imshow("Webcam", frame) # turns camera tab on

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

