# Soccer-Player-Re-identification with YOLO

This project implements a grid-based tracking system for football players using YOLOv8 for detection. It identifies and assigns consistent player IDs throughout a video clip, even in challenging conditions like overlaps and motion.

## Project Structure
├── best.pt   # Trained YOLOv8 model
├── 15sec_input_720p.mp4   # Input video file
├── output_tracked_video.mp4   # Output video with tracked players
├── FPRI.py   # Python script to run the model
└── README.md   # Documentation

## Key Features

- Player detection using a pretrained YOLOv8 model.
  
- Grid-based spatial re-identification logic.
  
- Colour-coded bounding boxes in output video.


## How to Run

1. **Clone the repository or download the files.**
2. **Place your video file and model (`best.pt`) in the same directory** as the script.
3. **Install dependencies** (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt

## To do manually:
pip install ultralytics opencv-python numpy

## Update paths in FPRI.py if needed:

model = YOLO("path/to/best.pt")

cap = cv2.VideoCapture("path/to/input_video.mp4")


## Run the script:

python FPRI.py


## Dependencies

Library         	Version (Recommended)

Python	     >=        3.8

OpenCV	     >=        4.5

NumPy	       >=        1.20

Ultralytics  >=        8.0 (YOLOv8)


## You can install them all using:
pip install ultralytics opencv-python numpy

## Parameters in Script

-GRID_SIZE = 1000: Used to create a 1000x1000 grid overlay for spatial tracking.

-MAX_MISSING_FRAMES = 20: Player ID is retained for 20 missing frames.

-POSITION_THRESHOLD = 50: Grid-cell distance threshold for re-identification.

## The script outputs:

A video (output_tracked_video.mp4) with bounding boxes and IDs for each player.
Real-time display of tracking (cv2.imshow).

Notes
Only the "Player" class (class ID 2) is tracked with IDs.
Other classes (Ball, Goalkeeper, Referee) are detected but not tracked.
