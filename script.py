import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui

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

#draw landmarks

def draw_landmarks(frame, hand_landmarks):
    h, w, _ = frame.shape
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (int(cx), int(cy)), 3, (255, 0, 0), -1)

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

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        draw_landmarks(frame, hand)

        index_tip = hand[8]
        screen_x, screen_y = map_to_screen(index_tip.x, index_tip.y, screen_w, screen_h)
        #debug prints
        print(screen_x, screen_y)
        print("Hand detected:", result.hand_landmarks[0][8])

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()