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
# smoothening values
smoothing = 0.3
prev_x, prev_y = screen_w // 2, screen_h // 2

#middle finger
click_range = 0
min_click_range = click_range-10
max_click_range = click_range+10
# pinky+thumb rightclick
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
        # draw_landmarks(frame, hand) # mark all landmarks
        index_tip = hand[8]
        middle_tip = hand[12]
        thumb_tip = hand[4]
        pinky_tip = hand[20]
        #click detect
        h, w, _ = frame.shape
        # finger position y
        iy = int(index_tip.y * h)
        my = int(middle_tip.y  * h)
        py = int(pinky_tip.y * h)
        ty = int(thumb_tip.y * h)
        y_diff_rc = ty - py
        y_diff_lc = my - iy

        # finger position x
        ix = int(index_tip.x * w)
        mx = int(middle_tip.x * w)
        px = int(pinky_tip.x * w)
        tx = int(thumb_tip.x * w)
        x_diff_rc = tx - px
        x_diff_lc = mx - ix

        if min_click_range <= y_diff_lc <= max_click_range:
            if min_click_range <= x_diff_lc <= max_click_range:
                pyautogui.leftClick()
               #debug print for left click
                # print("left click detected")
        elif min_click_range_rc <= y_diff_rc <= max_click_range_rc:
            if min_click_range_rc <= x_diff_rc <= max_click_range_rc:
                pyautogui.rightClick()
                # debug print for right click
                # print("right click detected")
        screen_x, screen_y = map_to_screen(index_tip.x, index_tip.y, screen_w, screen_h)
        smooth_x = prev_x + (screen_x - prev_x) * (1 - smoothing)
        smooth_y = prev_y + (screen_y - prev_y) * (1 - smoothing)

        pyautogui.moveTo(smooth_x, smooth_y)
        prev_x, prev_y = smooth_x, smooth_y #ADD SMOOTHENING VIA MOVEMENT THRESHOLD
        #debug prints
        # print(screen_x, screen_y)
        # print("Hand detected:", result.hand_landmarks[0][8])

   # cv2.imshow("Webcam", frame) # turns camera tab on

cap.release()
cv2.destroyAllWindows()