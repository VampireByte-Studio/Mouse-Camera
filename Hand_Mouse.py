import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui

#ASL related values
ASL = False
keyboard_cd = 1
lastActionTime = 0
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
# smoothening values
smoothing = 0.3
prev_x, prev_y = screen_w // 2, screen_h // 2
#range values
#leftflick range
click_range = 0
min_click_range = click_range-10
max_click_range = click_range+10
#rightclick range
right_click_range = 10
min_click_range_rc = right_click_range-10
max_click_range_rc = right_click_range+10
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

#map landmarks

def map_to_screen(norm_x, norm_y, screen_w, screen_h):
    screen_x = int(norm_x * screen_w)
    screen_y = int(norm_y * screen_h)
    return screen_x, screen_y

# mapping tip detection

def contact(tipA, tipB, center_range, w, h, tolerance):
    ax, ay = int(tipA.x * w), int(tipA.y * h)
    bx, by = int(tipB.x * w), int(tipB.y * h)
    x_diff = bx - ax
    y_diff = by - ay
    min_range = center_range - tolerance
    max_range = center_range + tolerance
    return min_range <= x_diff <= max_range and min_range <= y_diff <= max_range




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
        middle_tip = hand[12]
        ring_tip = hand[16]
        thumb_tip = hand[4]
        pinky_tip = hand[20]

        h, w, _ = frame.shape
        # ... your y_diff / x_diff calcs if still needed ...

        # mouse mode
        if not ASL:
            if contact(index_tip, middle_tip, click_range, w, h, 10):
                pyautogui.leftClick()
                lastActionTime = now
                print("Left click detected")
            elif contact(thumb_tip, pinky_tip, right_click_range, w, h, 10):
                pyautogui.rightClick()
                lastActionTime = now
                print("Right click detected")
            elif now - lastActionTime > keyboard_cd and contact(thumb_tip, ring_tip, click_range, w, h, 10):
                ASL = True
                print("KEYBOARD ON")
                lastActionTime = now
            elif contact(thumb_tip, middle_tip, click_range, w, h, 10):
                break

            screen_x, screen_y = map_to_screen(index_tip.x, index_tip.y, screen_w, screen_h)
            smooth_x = prev_x + (screen_x - prev_x) * (1 - smoothing)
            smooth_y = prev_y + (screen_y - prev_y) * (1 - smoothing)
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y

        if ASL:
            now = time.time()
            if now - lastActionTime > keyboard_cd and contact(thumb_tip, ring_tip, click_range, w, h, 10):
                ASL = False
                print("KEYBOARD OFF")
                lastActionTime = now


    cv2.imshow("Webcam", frame) # turns camera tab on

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()