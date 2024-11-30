import os

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Task2

from task2.corners_detection import *
from task2.corners_tracking2 import *
from task2.stickers_detection import *
from task2.chessboard_homography import *
from task2.utils import resize_frame

# Task 3

from task3.chessboard_analysis import *
from task3.movement_analysis import *
from task3.actualize_game_state import *

from task3.chessboard_utils import *

# Task 4

from task4.display_chess_game import *
from task4.display_chess_3d import *

import trimesh



def process_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        print(f"Error code: {cap.get(cv2.CAP_PROP_FOURCC)}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Frame parameters
    frame_count = 0
    frame_interval = 25
    frame_save_interval = 50
    
    # Cache to store the last valid detections
    cache = {
        'frame': None,

        'last_frame':None,
        'extended_grid': None,
        'extended_mask': None,

        'H': None,
        'objp': None,
        'rvec': None,
        'tvec': None,

        'chessboard_corners_extremities': None,
        'blue_stickers': None,
        'pink_stickers': None,
        'labeled_corners': None,
    }

    # vertices, faces = load_chess_piece()

    last_game_state = None
    actualized_game_state = {}
    potential_castling = None

    skip_moment = False

    # Ajout des variables de contrôle
    paused = False
    frame_step = 0  # Pour stocker le nombre de frames à avancer/reculer

    # Dictionnaire pour stocker l'historique des états
    game_history = {}  # frame_count sera la clé principale
    last_frame_analyzed = 0

    while True:
        if not paused:
            frame_count += 1
        else:
            frame_count += frame_step
        
        frame_step = 0  # Reset le step après utilisation
        
        if frame_count < 0:  # Empêcher d'aller avant le début
            frame_count = 0
            
        if frame_count % frame_interval != 0:
            continue

        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()

        if not ret:
            if frame_count > cap.get(cv2.CAP_PROP_FRAME_COUNT):  # Si on dépasse la fin
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
            if not ret:  # Si toujours pas de frame valide
                break


        # ! TASK 2 : Corners detection

        ###########################################
        # * STICKERS DETECTION
        ###########################################

        # Detect colored stickers and label corners
        blue_stickers, pink_stickers = detect_stickers(frame)
        

        ###########################################
        # * CORNERS DETECTION AND TRACKING
        ###########################################

        chessboard_corners = detect_corners(frame)
        chessboard_corners_extremities = None

        # Detect chessboard corners
        if chessboard_corners is not None: # * Found corners with cv2.findChessboardCorners

            # chessboard_corners_extremities = detect_chessboard_corners_extremities(frame, chessboard_corners)
            extended_grid, chessboard_corners_extremities = detect_all_chessboard_corners(frame, chessboard_corners)

            if chessboard_corners_extremities: # * SAVE DETECTION INFORMATIONS
                cache['chessboard_corners_extremities'] = chessboard_corners_extremities
                cache['extended_mask'] = np.ones((9, 9), dtype=bool)
                cache['extended_grid'] = extended_grid
                cache['last_frame'] = frame


        else: # * No corners found with cv2.findChessboardCorners

            chessboard_corners_extremities = cache['chessboard_corners_extremities']
            
            # * SEARCH CORNERS USING LAST KNOWN CORNERS AND CORNERS TRACKING

            if cache['extended_grid'] is not None and cache['last_frame'] is not None:

                extremities, grid, mask = estimate_corners_movement(cache['extended_grid'], cache['extended_mask'], frame, cache['last_frame'], debug=False)

                cache['extended_grid'] = grid
                cache['extended_mask'] = mask
                cache['last_frame'] = frame

                if extremities is not None:
                    chessboard_corners_extremities = extremities


        # Conversion for [array(,)] to [(,)]
        chessboard_corners_extremities = [tuple(corner) for corner in chessboard_corners_extremities] if chessboard_corners_extremities is not None else []
        cache['chessboard_corners_extremities'] = chessboard_corners_extremities


        ###########################################
        # * CORNER LABELING
        ###########################################
        
        # TODO : label corners without using stickers

        # labeled_corners = None
        labeled_corners = label_corners(chessboard_corners_extremities, blue_stickers, pink_stickers)

        # Update cache only with valid detections
        cache['blue_stickers'] = blue_stickers if blue_stickers else cache['blue_stickers']
        cache['pink_stickers'] = pink_stickers if pink_stickers else cache['pink_stickers']

        if labeled_corners is not None:
            cache['labeled_corners'] = labeled_corners if all(corner is not None for corner in labeled_corners.values()) else cache['labeled_corners']


        


        ###########################################
        # * VISUALIZATION
        ###########################################

        warped_frame = None

        if cache['labeled_corners'] is not None and all(corner is not None for corner in cache['labeled_corners'].values()):
            warped_frame = get_warped_image(frame, cache['labeled_corners'])
            frame = draw_labeled_chessboard(frame, cache['labeled_corners'])

        if cache['chessboard_corners_extremities'] is not None and chessboard_corners_extremities is not None:
            # frame = draw_refined_corners(frame, cache['chessboard_corners_extremities'], chessboard_corners_refined, search_radius=radius)
            frame = draw_corners(frame, chessboard_corners_extremities)

        if cache['blue_stickers'] is not None and cache['pink_stickers'] is not None:
            frame = draw_stickers(frame, cache['blue_stickers'], cache['pink_stickers'])

        


        # ! Task 3 : Game analysis

        print(f"\nCurrent frame: {frame_count}")

        # print(f"Frames in history: {list(game_history.keys())}")

        if frame_count % frame_save_interval == 0:
            #########################################
            # * CHESSBOARD ANALYSIS
            #########################################

            # * Si on revient en arrière et qu'on a déjà analysé cette frame
            if frame_count in game_history.keys():
                print('USING HISTORY')
                img_display = game_history[frame_count]['display_frame']
                cv2.imshow('warped_frame', img_display)

            else:
                print('NEW ANALYSIS')
                # * Analyse normale

                # last_frame_analyzed = frame_count
                square_results, img, filtered_images, stats = analyze_chess_board(warped_frame)

                # Retrieve datas
                game_state = retrieve_game_state(square_results, last_game_state)

                img_display = display_game_state(square_results, stats, img, filtered_images)


                state = {
                    'frame': frame_count,
                    'gs': game_state
                }


                #########################################
                # * GAME STATE ANALYSIS
                #########################################

                curr_state = np.fliplr(game_state.copy())

                if last_game_state is None and curr_state is not None:

                    # * First state found --> initialization
                    actualized_game_state = initialize_game_state(curr_state)
                    last_game_state = curr_state

                
                elif last_game_state is not None and game_state is not None:

                    # * Update game state
                    print('MOVE ANALYSIS')
                    move_analysis, potential_castling = analyze_move(last_game_state, curr_state, potential_castling, actualized_game_state)

                    # * If move is valid
                    if move_analysis['valid']:

                        # * Move is valid with error correction
                        if move_analysis['move_type'] != 'castling' and move_analysis['error_pos'] is not None:

                            error_pos = move_analysis['error_pos']
                            print('Correct error position : ', error_pos)
                            curr_state[error_pos] = last_game_state[error_pos]

                        last_game_state = curr_state.copy()

                        # * If move is castling
                        if move_analysis['move_type'] == 'castling':

                            actualized_game_state, board, piece_certainty = actualize_game_state_with_castling(actualized_game_state, move_analysis, curr_state)

                        else:
                            # * Move is valid 
                            actualized_game_state, board, piece_certainty = actualize_game_state(actualized_game_state, move_analysis, curr_state)
                            
                            from_pos = move_analysis['from_pos']
                            to_pos = move_analysis['to_pos']
                            
                            # Convertir les positions en notation d'échecs
                            from_square = f"{chr(97 + from_pos[1])}{8 - from_pos[0]}"
                            to_square = f"{chr(97 + to_pos[1])}{8 - to_pos[0]}"
                            
                            if move_analysis['move_type'] == 'move':

                                print(f"Move: {from_square} -> {to_square}")
                            
                            else:  # capture
                                print(f"Capture: {from_square} x {to_square}")


                        # print(piece_certainty)
                        # draw_chessboard(board, show=True)

                        img_display = display_chess_game_2d(img_display, actualized_game_state)
                        cv2.imshow('warped_frame', img_display)

                        # Sauvegarder l'état actuel dans l'historique
                        


                game_history[frame_count] = {
                    'display_frame': img_display.copy(),
                    'game_state': curr_state.copy() if curr_state is not None else None,
                    'actualized_state': actualized_game_state.copy() if actualized_game_state else None
                }
                print(f"Saved frame {frame_count} to history")



            # data['game_states'].append(state)
            # last_game_state = curr_state


        


        ###########################################
        # * HOMOGRAPHY AND POSE ESTIMATION
        ###########################################
        
        # Compute homography matrix
        H = None
        objp = None

        if chessboard_corners is not None:
            H, objp = compute_homography(chessboard_corners)
        
        # Update cache with valid homography results
        if H is not None:
            cache['H'] = H
            cache['objp'] = objp

        # Compute and draw 3D pose
        if cache['labeled_corners'] is not None and cache['objp'] is not None:
            rvec, tvec = compute_pose(cache['objp'], cache['labeled_corners'])
            cache['rvec'] = rvec
            cache['tvec'] = tvec

        # Camera parameters
        calibration_results = np.load('camera_calibration_results.npz')
        camera_matrix = calibration_results['cameraMatrix']
        dist_coeffs = calibration_results['distCoeffs']


        if cache['rvec'] is not None and cache['tvec'] is not None:

            params = {
                'rvec': cache['rvec'],
                'tvec': cache['tvec'],
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs
            }


            # frame = display_chess_piece(frame, params, 5, 5)

            

            frame = display_chess_game_3d(frame, params, actualized_game_state)

            # Garder l'affichage des axes si souhaité
            frame = draw_axis(frame, cache['rvec'], cache['tvec'])

        ###########################################
        # * DISPLAY RESULTS
        ###########################################

        frame = resize_frame(frame, 1200)
        cv2.imshow('frame', frame)

        


        ##########################################
        # * CONTROLS
        ##########################################

        key = cv2.waitKey(1 if not paused else 0)  # Attend indéfiniment si en pause
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord(' '):  # Espace pour pause/play
            paused = not paused
            print('Pause' if paused else 'Play')
        elif paused:  # Ces contrôles ne fonctionnent qu'en pause
            if key & 0xFF == ord('s'):  # s pour reculer
                print('Previous frame')
                frame_step = -frame_interval
            elif key & 0xFF == ord('d'):  # d pour avancer
                print('Next frame')
                frame_step = frame_interval


        ###########################################
        # * SAVE RESULTS
        ###########################################
        
        # Save every 100th frame
        # os.makedirs('images_1', exist_ok=True)
        # if frame_count % frame_save_interval == 0 or skip_moment:
        #     if frame is not None:
        #         frame_name = f"frame_{frame_count:06d}.png"
        #         cv2.imwrite(os.path.join('images_results/frames', frame_name), frame)
        #         print(f"Saved: {frame_name}")

        #     if warped_frame is not None:
        #         warped_frame_name = f"warped_frame_{frame_count:06d}.png"
        #         cv2.imwrite(os.path.join('images_results/warped_images', warped_frame_name), warped_frame)
        #         print(f"Saved: {warped_frame_name}")

        #     skip_moment = False

        # frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    video_path = ('C:/Users/VaryaStrizh/CV/elen0016-computer-vision-tutorial-master/elen0016-computer-vision-tutorial'
                  '-master/project/task2/videos/moving_2.mov')  # Remplacez par le chemin de votre vidéo
    

    video_path = 'videos/moving_game.mov'

    process_video(video_path)
