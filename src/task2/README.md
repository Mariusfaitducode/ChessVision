# ELEN-0016 – Computer Vision: Task 2 - Camera Calibration

**Authors**: Varvara Strizhneva, Marius Diguat

---

## Overview

In this task, it was necessary to implement techniques for calibrating a chessboard and detecting its corners and stickers using video input. The goal is to identify the sides of the chessboard in the fixed and moving positions.

We then compute and draw the polygon formed by the detected corners, saving annotated images every 100th frame.

---

## Project Structure

The provided `.zip` archive includes the following files:

- `camera_calibration.py`: Script for calibrating the camera using the input video.
- `video_analysis.py`: Main script for processing the video, detecting chessboard corners and stickers, computing the homography matrix, and saving results, which is based on other
scprits.
- `corners_detection.py`: Functions for detecting chessboard corners.
- `stickers_detection.py`: This function detects blue and pink stickers on the chessboard.
- `chessboard_homography.py`: Functions for computing the homography of the detected corners.
- `utils.py`: Some utility functions used across the project.

---

## Instructions for Use

### Step 1: Camera Calibration

1. Open `camera_calibration.py`.
2. Specify your video path at the end of the script.
3. Run the script. It will generate a file named `camera_calibration_results.npz` containing the calibration parameters. These are required for video analysis file.

### Step 2: Video Analysis and Corner Detection

1. Open `video_analysis.py`.
2. Set your video path in the script.
3. Run the script. The processed images with detected corners and stickers will be saved every 100th frame in the `images` directory. 
   - **Note**: Not for every frame it was possible to detect the frames, so we used cache to save the result from previous frames for the current frame.
   - **Note**: To look at a complete video with detected results, uncomment the relevant section in the `process_video` function.

### Running Individual Modules

- You can run `corners_detection.py` and `chessboard_homography.py` separately to test and verify the corner detection and homography algorithms.

---

## The reason for chosen model for calibration

For the calibration the findChessboardCorners function was chosen due to several reasons
The `findChessboardCorners` function is part of the OpenCV library, because of this
it is relatively straightforward to integrate into our project without needing to implement a custom corner detection algorithm from scratch.
Moreover, `findChessboardCorners` integrates with other OpenCV functions used for camera calibration, such as `calibrateCamera` and `cornerSubPix`. 
