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

- 
