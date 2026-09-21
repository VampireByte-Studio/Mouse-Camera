# 🖱️ Camera Mouse

A Python project that lets you control your computer's mouse using hand gestures detected through a webcam.

Instead of using a physical mouse, the project uses computer vision to track your hand and translate its movements and gestures into mouse actions.

## ✨ Features

🖐️ Real-time hand tracking through a webcam

🖱️ Move the mouse using your hand

👆 Gesture-based clicking

✌️ Support for multiple hand gestures

🎥 Live camera feed

⚡ Real-time gesture detection

🧠 Uses computer vision to identify hand landmarks

## 🛠️ Technologies

This project is built using:

Python

OpenCV — camera input and image processing

MediaPipe — hand tracking and landmark detection

PyAutoGUI — controlling the computer mouse

## 📋 Requirements

Make sure you have Python installed on your computer.

Install the required packages with:

pip install opencv-python mediapipe pyautogui

## 🚀 Getting Started
1. Clone the repository
git clone https://github.com/VampireByte-Studio/Mouse-Camera.git

2.
pip install opencv-python mediapipe pyautogui

3. Run the project


Your webcam should open and begin tracking your hand.

## 🎮 Controls + 🧠 How It Works

You can modify the gesture detection code to create your own controls.


The project follows a simple pipeline:

Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Tracking
   ↓
Hand Landmark Detection
   ↓
Gesture Recognition
   ↓
Mouse Movement / Click


MediaPipe detects key points on the hand, such as the fingertips and joints.

The program then uses those landmarks to determine the user's hand position and gesture.

The pointer fingertip can be used as the mouse cursor position:

Index Finger
     ↓
Hand Landmark
     ↓
Screen Coordinates
     ↓
Mouse Position

To left click overlap the Middle finger with the Point Finger (GREEN OVER BLUE DOTS)
and for Right click overlap the Thumb finger over Pinky finger (Yellow OVER YELLOW DOTS)

## 📍 Hand Landmarks

MediaPipe provides 21 hand landmarks. Some of the landmarks used by this project include:

4 — Thumb tip

8 — Index finger tip

12 — Middle finger tip

20 — Pinky tip

These landmarks can be used to detect different gestures.

## ⚠️ Notes

Because this project relies on a webcam and real-time computer vision, performance can depend on:

Camera quality

Lighting conditions

CPU/GPU performance

Distance between your hand and camera

Background complexity

Good lighting and a clear view of your hand generally produce better tracking.


more updates coming soon such as a ASL Keyboard

## 🔒 Privacy

This project processes the webcam feed locally for hand tracking. It does not need to upload your camera footage to an external server.

Make sure you understand and trust any dependencies you install before running the project.

## 📄 License

This project is intended for personal, educational, and non-profit use only.

You may use, copy, modify, and distribute this code for projects that are not intended to generate monetary profit.

Commercial use is prohibited without prior written permission from the copyright holder.

If you would like to use this project commercially, please contact the copyright holder for permission.

