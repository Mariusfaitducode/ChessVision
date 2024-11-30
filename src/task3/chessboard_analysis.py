import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from skimage.filters import sobel

from case_analysis import detect_if_case_is_occupied, detect_piece_color
from case_color_analysis import classify_pieces



def analyze_chess_board(frame):
    # Read the image
    img = frame

    ###########################################
    # * FILTERING
    ###########################################
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    ###########################################
    # * BOARD DATAS
    ###########################################
    
    # Chessboard dimensions (8x8)
    rows, cols = 8, 8
    
    # Get image dimensions
    height, width = gray.shape
    
    # Calculate square size
    square_h = height // rows
    square_w = width // cols
    
    # Dictionary to store results
    square_results = {}
    square_stats = {}
    
    # Define margin (as percentage of square size)
    margin_percent = 0.15  # 10% margin
    
    index = 0
    
    ###########################################
    # * ANALYSIS
    ###########################################

    occupied_squares = []

    # Process each square
    for i in range(rows):
        for j in range(cols):

            square_name = f"{chr(72-i)}{8-j}"

            # Full square coordinates
            top = i * square_h
            left = j * square_w
            bottom = (i + 1) * square_h
            right = (j + 1) * square_w

            # Calculate margins in pixels
            margin_h = int(square_h * margin_percent)
            margin_w = int(square_w * margin_percent)
            
            # Analysis zone coordinates (with margins)
            inner_top = top + margin_h
            inner_left = left + margin_w
            inner_bottom = bottom - margin_h
            inner_right = right - margin_w
            
            is_occupied, edge_percentage, pixel_variance = detect_if_case_is_occupied(edges, blurred, inner_top, inner_left, inner_bottom, inner_right)

            piece_color = None
            piece_peak = None
            
            if is_occupied:

                # Convert to RGB Image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Extract the inner square region for color analysis
                piece_region = gray[inner_top:inner_bottom, inner_left:inner_right]
                
                # Determine if square is dark or light based on position
                is_dark_square = (i + j) % 2 == 0

                occupied_squares.append((piece_region, is_dark_square, square_name))
                
                # piece_color = detect_piece_color(piece_region, is_dark_square)
            # square_name = f"{chr(65+j)}{8-i}"

            square_results[square_name] = {
                'is_occupied': is_occupied,
                'piece_color': piece_color
            }
            square_stats[square_name] = {
                'edge_percentage': edge_percentage,
                'pixel_variance': pixel_variance,
                'index': index,
                'name': square_name,
                'piece_peak': piece_peak
            }
            
            index += 1

            # For visualization, also draw the analyzed inner area
            
            img = cv2.rectangle(img, 
                        (inner_left, inner_top), 
                        (inner_right, inner_bottom), 
                        (128, 128, 128), 1)  # Gray rectangle to show analyzed area
            
    if len(occupied_squares) > 0:
        piece_colors = classify_pieces(occupied_squares, debug=True)

    # Mettre à jour square_results avec les couleurs des pièces
    for square_name, piece_info in square_results.items():
        if piece_info['is_occupied']:
            piece_info['piece_color'] = piece_colors.get(square_name, None)
    
    return square_results, img, {
        'gray': gray,
        'blurred': blurred,
        'edges': edges,
    }, square_stats



def analyze_all_images(folder_path):
    results = {}
    
    # Process all images in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):

            image_path = os.path.join(folder_path, filename)

            frame = cv2.imread(image_path)

            square_results, img, filtered_images, stats = analyze_chess_board(frame)
            results[filename] = {
                'square_results': square_results,
                'image': img,
                'filtered': filtered_images,
                'stats': stats
            }
    
    # Interactive visualization
    image_names = list(results.keys())
    current_idx = 0
    current_view = 'original'  # 'original', 'gray', 'blurred', 'edges'

    data = {}
    data['game_states'] = []

    ###########################################
    # * RETRIEVE GAME STATES
    ###########################################

    last_game_state = None

    for i in range(len(image_names)):
        # data['game_states'].append(None)

        current_image = image_names[i]
        result = results[current_image]
        
        # Retrieve datas

        game_state = np.zeros((8, 8), dtype=int)

        for i in range(8):
            for j in range(8):
                square_name = f"{chr(65+j)}{8-i}"
                square_result = result['square_results'][square_name]

                is_occupied = square_result['is_occupied']

                piece_color = square_result['piece_color']

                # if current_image == 'warped_frame_000800.png':
                #     print(square_name, result['square_results'][square_name])

                if is_occupied is True:

                    if piece_color is not None:

                        if piece_color == 'black':
                            game_state[i, j] = -7
                        else:
                            game_state[i, j] = 7

                    elif last_game_state is not None:
                        game_state[i, j] = last_game_state[i, j]

        # if current_image == 'warped_frame_000800.png':
            # print(result['square_results'][square_name])
            # print(game_state)

        state = {
            'frame': current_image,
            'gs': game_state.tolist()
        }

        data['game_states'].append(state)

        last_game_state = game_state

    with open('game_state.json', 'w') as f:
        json.dump(data, f)


    ###########################################
    # * VISUALIZE RESULTS
    ###########################################

    while True:
        current_image = image_names[current_idx]
        result = results[current_image]
        
        # Select image to display based on current view
        if current_view == 'original':
            img_display = result['image'].copy()
        else:
            img_display = cv2.cvtColor(result['filtered'][current_view], cv2.COLOR_GRAY2BGR)
        
        # Chessboard dimensions
        height, width = img_display.shape[:2]
        square_h = height // 8
        square_w = width // 8
        
        # Draw rectangles and add text
        for i in range(8):
            for j in range(8):
                top = i * square_h
                left = j * square_w
                bottom = (i + 1) * square_h - 2
                right = (j + 1) * square_w - 2
                
                square_name = f"{chr(72-i)}{8-j}"

                square_result = result['square_results'][square_name]

                is_occupied = square_result['is_occupied']
                piece_color = square_result['piece_color']
                
                stats = result['stats'][square_name]
                
                # Green for occupied squares, red for empty ones
                color = (0, 255, 0) if is_occupied else (0, 0, 255)
                cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
                
                piece_peak = stats['piece_peak']

                if piece_color is not None:
                    text = f"{piece_color}"
                    cv2.putText(img_display, text, (left + 5, top + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                    # Addtext
                text = f"{stats['edge_percentage']:.2f} -- {stats['pixel_variance']:.2f}%"
                cv2.putText(img_display, text, (left + 5, top + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                

        

        # with open('game_state.json', 'w') as f:
        #     json.dump(state, f)

        
                

        ###########################################
        # * DISPLAY
        ###########################################
        

        # Display image
        cv2.imshow('Chess Analysis', img_display)
        print(f"\nAnalyzing {current_image} - Mode: {current_view}")
        
        # Key handling
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key in [ord('d'), 83]:  # Right arrow
            current_idx = (current_idx + 1) % len(image_names)
        elif key in [ord('s'), 81]:  # Left arrow
            current_idx = (current_idx - 1) % len(image_names)
        elif key == ord('v'):  # Change view
            views = ['original', 'gray', 'blurred', 'edges']
            current_view = views[(views.index(current_view) + 1) % len(views)]
    
    cv2.destroyAllWindows()
    return results


if __name__ == "__main__":
    folder_path = "images_results/warped_images"
    results = analyze_all_images(folder_path)
