import task3.chessboard_utils as utils

import numpy as np

import cv2

# def get_piece_image(piece_name):

#     return utils.NUM_TO_PIECE[piece_name]


def display_chess_game_2d(img_display, game_actualization):

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

            certainties = game_actualization['piece_certainty'][(j, i)]

            CERTAINTY_THRESHOLD = 0.4
            for piece, prob in certainties.items():
                if prob > CERTAINTY_THRESHOLD:
                    piece_image = utils.piece_images[piece]
                    piece_image = piece_image.resize((square_w, square_h))

                    # Convert piece_image to numpy array with alpha channel
                    piece_array = np.array(piece_image.convert('RGBA'))
                    
                    # Convert BGR to RGB for proper color display
                    rgb = cv2.cvtColor(piece_array[:, :, :3], cv2.COLOR_BGR2RGB)
                    
                    # Get alpha channel
                    alpha = piece_array[:, :, 3] / 255.0
                    
                    # Create 3D alpha for broadcasting
                    alpha_3d = np.stack([alpha] * 3, axis=2)
                    
                    # Get the region of the background image
                    background = img_display[top:top+square_h, left:left+square_w]
                    
                    # Blend the piece with the background
                    blended = (rgb * alpha_3d + background * (1 - alpha_3d)).astype(np.uint8)
                    
                    # Update the image
                    img_display[top:top+square_h, left:left+square_w] = blended

    # Show the image
    cv2.imshow('Chess Game', img_display)
     
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # Espace
            break
        elif key == ord('q'):  # q
            cv2.destroyAllWindows()
            exit()

    return img_display

            
            
            # Green for occupied squares, red for empty ones
            # color = (0, 255, 0) if is_occupied else (0, 0, 255)
            # cv2.rectangle(img_display, (left, top), (right, bottom), color, 2)
            
            # piece_peak = stats['piece_peak']

            # if piece_color is not None:
            #     text = f"{piece_color}"
            #     cv2.putText(img_display, text, (left + 5, top + 20),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            #     # Addtext
            # text = f"{stats['edge_percentage']:.2f} -- {stats['pixel_variance']:.2f}%"
            # cv2.putText(img_display, text, (left + 5, top + 80),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
