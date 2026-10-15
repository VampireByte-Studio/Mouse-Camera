import json
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
OUTPUT_FILE = "signs.jsonl"
CAPTURE_INTERVAL = 0.10  # seconds between samples while recording


def create_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    return vision.HandLandmarker.create_from_options(options)


def normalize(landmarks):
    """Wrist to origin, scale by wrist -> middle-finger-base. Returns 63 numbers."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    pts -= pts[0]
    scale = np.linalg.norm(pts[9])
    if scale < 1e-6:
        return None
    pts /= scale
    return np.round(pts.flatten(), 5).tolist()


def save_sample(label, hand, vector):
    sample = {"label": label, "hand": hand, "landmarks": vector}
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(sample) + "\n")


def draw_points(frame, landmarks):
    h, w = frame.shape[:2]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 200, 255), -1)


def draw_hud(frame, label, recording):
    status = "Recording" if recording else "paused"
    color = (0, 0, 255) if recording else (0, 200, 0)
    cv2.putText(frame, f"Label: {label or '-'}  [{status}]", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    cap = cv2.VideoCapture(0)
    landmarker = create_landmarker()
    label = None
    recording = False
    last_capture = 0.0
    start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # flip BEFORE detection (handedness)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            hand = result.handedness[0][0].category_name  # "Left" / "Right"
            draw_points(frame, landmarks)

            now = time.time()
            if recording and label and now - last_capture >= CAPTURE_INTERVAL:
                vector = normalize(landmarks)
                if vector is not None:
                    save_sample(label, hand, vector)
                    last_capture = now

        draw_hud(frame, label, recording)
        cv2.imshow("Hand Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:        # ESC = quit
            break
        elif key == 32:      # SPACE = toggle recording
            recording = not recording
        elif key != 255 and chr(key).isalnum():
            label = chr(key).upper()   # A-Z / 0-9 = set label

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()