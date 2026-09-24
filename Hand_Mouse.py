import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui

#ASL related values
ASL = False
#time related values
click_cd = 0.4
type_cd = 0.2
mode_cd = 2
lastClickTime = 0
lastModeTime = 0
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

def contact_y(tipA, tipB, center_range, h, tolerance):
    ay = int(tipA.y * h)
    by = int(tipB.y * h)
    y_diff = by - ay
    min_range = center_range - tolerance
    max_range = center_range + tolerance
    return min_range <= y_diff <= max_range

def contact_x(tipA, tipB, center_range, w, tolerance):
    ax = int(tipA.x * w)
    bx = int(tipB.x * w)
    x_diff = bx - ax
    min_range = center_range - tolerance
    max_range = center_range + tolerance
    return min_range <= x_diff <= max_range

def distant_y(tipA, tipB, h, d):
    ay = int(tipA.y * h)
    by = int(tipB.y * h)
    y_diff = by - ay
    return y_diff > d

def distant_x(tipA, tipB, w, d):
    ax = int(tipA.x * w)
    bx = int(tipB.x * w)
    x_diff = bx - ax
    return x_diff > d

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
        #joint maps

        #finger tip alternatives
        index_tip = hand[8]
        middle_tip = hand[12]
        ring_tip = hand[16]
        thumb_tip = hand[4]
        pinky_tip = hand[20]

        #joint general mappings
        h1 = hand[1]
        h2 = hand[2]
        h3 = hand[3]
        h4 = hand[4]
        h5 = hand[5]
        h6 = hand[6]
        h7 = hand[7]
        h8 = hand[8]
        h9 = hand[9]
        h10 = hand[10]
        h11 = hand[11]
        h12 = hand[12]
        h13 = hand[13]
        h14 = hand[14]
        h15 = hand[15]
        h16 = hand[16]
        h17 = hand[17]
        h18 = hand[18]
        h19 = hand[19]
        h20 = hand[20]


        h, w, _ = frame.shape
        # ... your y_diff / x_diff calcs if still needed ...

        # mouse mode
        if not ASL:
            now = time.time()
            if now - lastClickTime > click_cd and contact(index_tip, middle_tip, click_range, w, h, 10):
                pyautogui.leftClick()
                lastClickTime = now
                print("Left click detected")
            elif now - lastClickTime > click_cd and contact(thumb_tip, pinky_tip, right_click_range, w, h, 10):
                pyautogui.rightClick()
                lastClickTime = now
                print("Right click detected")
            elif now - lastModeTime > mode_cd and contact(thumb_tip, ring_tip, click_range, w, h, 10):
                ASL = True
                print("KEYBOARD ON")
                lastModeTime = now
            #elif contact(thumb_tip, middle_tip, click_range, w, h, 10):
                #break

            screen_x, screen_y = map_to_screen(index_tip.x, index_tip.y, screen_w, screen_h)
            smooth_x = prev_x + (screen_x - prev_x) * (1 - smoothing)
            smooth_y = prev_y + (screen_y - prev_y) * (1 - smoothing)
            pyautogui.moveTo(smooth_x, smooth_y)
            prev_x, prev_y = smooth_x, smooth_y


        #ASL KEYBOARD
        if ASL:
            now = time.time()
            if now - lastModeTime > mode_cd and contact(thumb_tip, ring_tip, click_range, w, h, 10):
                ASL = False
                print("KEYBOARD OFF")
                lastModeTime = now
            elif now - lastModeTime > type_cd and contact(h16, h13, click_range, w, h, 30) and contact(h9,h12, click_range, w, h, 30) and contact(h8, h5, click_range, w, h, 30):
                print ("A")

            elif now - lastModeTime > type_cd and contact(h4, h9, click_range, w, h, 10) and contact_y(h8, h16, click_range, h, 10) and distant_y(h12, h9, h, 60):
                print("B")
            elif now - lastModeTime > type_cd and contact(h8, h12, click_range, w, h, 10) and contact_x(h4, h8, click_range, w, 10) and distant_y(h4, h8, h, 20):
                print("C")
                #NOT WORKING


    cv2.imshow("Webcam", frame) # turns camera tab on

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

