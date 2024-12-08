# Introduction
In the following code, we provide a summary of the project structure and instructions for using the code.
The code can be run separately for each task, and for running the whole project, it's necessary to use `demonstration.py`

# Task 2
In this task, it was necessary to implement techniques for calibrating a chessboard and detecting its corners and stickers using video input. The goal is to identify the sides of the chessboard in the fixed and moving positions.

We then compute and draw 4 corners and annotations to them and detected stickers which help to identify the side of the board and label the corners, saving annotated images every 25th frame.

To do this, we used the functions provided in the project. To analyze the video, we used the video_analysis.py file, which takes as an input the path to the video, and output the images and wrapped images.

To get the results of images with corners it's necessary to run camera_calibration.py to calibrate the camera, and then video_analysis.py to process the video. To do so, it's required to write the path in both files to the video.
### Code Structure
At first, we use the function `detect_stickers` (`stickers_detection.py`) to detect the stickers, which is based on the color thresholding and masking the image with the corresponding range of colors. We then `draw_stickers` to draw the stickers on the chessboard
Then, we use the function `detect_corners` to detect the corners of the chessboard, which is based on the `detect_chessboard_corners` function, and refine them using `cv2.cornerSubPix()`, which finds the sub-pixel accurate location of the corners.
To detect all corners of a chessboard (8x8 squares) from the detected inner corners we use `detect_all_chessboard_corners`. It takes an image, detects inner corners, and the inner board size as input, and returns the coordinates of the 4 corners of each square and the 4 extreme corners of the chessboard.
It does this by extrapolating the outer edges of the chessboard based on the detected inner corners, and then calculating the coordinates of the corners of each square. That all can be seen in the `corners_detection.py` file.
Then, to fix the corners within the movement for moving videos, there was used `estimate_corners_movement`, which estimates the movement of the corners of a chessboard between two frames of a video.
Which returns updated corners and a new grid. Then we used `label_corners` to label the corners of the chessboard, and finally computed the homography matrix.

# Task 3
In this task we needed to take our results from the previous task, in particular wrapped images, and recognize piece colors and the name of the chess figure based on the movement of the figures.

To run the code it's necessary to use `chessboard_analysis.py` with the wrapped images.
The output is a .json file with the frame number and corresponding movement of the pieces on the board for this frame. To actualize the movements it's necessary to use `game_analysis.ipynb`.

To get the same result without running `game_analysis.ipynb` file separately, it's possible to use `demonstration.py` to run the whole project with the json file as an output.
In case of moving_video №1 to get the same results, it's important to discard condition (if len(positions) = 4) in `movement_analysis.py` to get the same results. 

### Code Structure
For each image we compute the chessboard, then for each square we analyze if the square is occupied by a piece using the `detect_if_case_is_occupied` function
Then if a square is occupied, the function extracts the inner square region and determines the piece color using the `classify_pieces`, which uses Gaussian Mixture to determine the probability of each pixel belonging to one of the 4 distributions of colors for each piece.
Then for each result from `analyze_chess_board` we use function `retrieve_game_state` that converts a dictionary of square results (square_results) into a 2D array (game_state) representing a chessboard.
After that we can use `analyze_all_images` to analyze all images in a folder, and the result of game states with position and color of the piece will be written in .json file.
To actualize the positions with the chessfigure, it's necessary to open `game_analysis.ipynb` file. Using functions from `actualize_game_state.py` and `movement_analysis.py` we can get the corresponding figures for this frame.

# Task 4
This part allows to augment the chess figures on the board in real-time.
`display_chess_3d.py` is a code for the chessboard visualization in 3D space, and `display_chess_2d.py`, which displays a 2D chess game board with pieces and their corresponding probabilities
# Result Code
To have the output .json file in one file, it can be used by running `demonstration.py`, which covers tasks 2, 3, and 4 in one file.
To do so, it's necessary to provide a path to the video in the `video_path` variable.
