import cv2
import numpy as np


sticker_history = {'blue': None, 'pink': None}
labeled_corners = {'a1': None, 'a8': None, 'h8': None, 'h1': None}


def detect_stickers(img, corners, distance_threshold=150):
    """
    Détecte les autocollants bleus et roses sur l'image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

    # range for blue sticker
    lower_blue = np.array([90, 98, 182])  # Lower bound in HSV
    upper_blue = np.array([100, 198, 248])

    # range for pink sticker
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])

    # threshold to get only blue and pink colors
    mask_blue = cv2.inRange(blurred, lower_blue, upper_blue)
    # kernel = np.ones((5, 5), np.uint8)
    # mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
    # mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

    mask_pink = cv2.inRange(blurred, lower_pink, upper_pink)

    pink_stickers = None
    previous_blue_sticker = None


    global sticker_history
    global labeled_corners

    # Detect blue stickers
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)

    current_blue_sticker = None

    if contours_blue:
        for c in contours_blue:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            current_blue_sticker = (int(cX), int(cY), int(radius))
            break
    else:
        if previous_blue_sticker is not None:
            blue_stickers = previous_blue_sticker


    # Update sticker history or use previous position if no new sticker detected
    if current_blue_sticker is not None:
        if sticker_history['blue'] is not None:
            dist = np.linalg.norm(np.array(current_blue_sticker) - np.array(sticker_history['blue']))
            if dist>distance_threshold:
                blue_stickers = sticker_history['blue']
            else:
                sticker_history['blue'] = current_blue_sticker
                blue_stickers = current_blue_sticker
        else:
            # No history; initialize with current
            blue_stickers = current_blue_sticker
            sticker_history['blue'] = current_blue_sticker
    else:
        # If no new blue sticker is detected, use the last known one
        if sticker_history['blue'] is not None:
            blue_stickers = sticker_history['blue']

    

    # Detect pink stickers
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink = sorted(contours_pink, key=cv2.contourArea, reverse=True)

    if contours_pink:
        ((cX, cY), radius) = cv2.minEnclosingCircle(contours_pink[0])
        # sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
        pink_stickers = (int(cX), int(cY), int(radius))


    
    #         break

    return blue_stickers, pink_stickers


def label_corners(corners, blue_sticker, pink_sticker):

    if blue_sticker is not None:
        # Trouve a1 (plus proche du sticker bleu)
        cX, cY, radius = blue_sticker
        sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
        labeled_corners['a1'] = tuple(sorted_corners[0])

    if pink_sticker is not None:
        # Trouve h1 (plus proche du sticker rose)
        cX, cY, radius = pink_sticker
        sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
        labeled_corners['a8'] = tuple(sorted_corners[0])



    if labeled_corners['a1'] is not None and labeled_corners['a8'] is not None:
        # Calcule le vecteur de la rangée 1 (de a1 à h1)
        rank_vector = np.array(labeled_corners['a8']) - np.array(labeled_corners['a1'])
        
        # Trouve les deux coins restants (les plus éloignés de la rangée 1)
        remaining_corners = [corner for corner in corners 
                           if corner != labeled_corners['a1'] and corner != labeled_corners['a8']]
        
        # Trie les coins restants par leur distance perpendiculaire à la rangée 1
        def perpendicular_distance(point):
            point_vector = np.array(point) - np.array(labeled_corners['a1'])
            projection = np.dot(point_vector, rank_vector) / np.dot(rank_vector, rank_vector)
            perpendicular = point_vector - projection * rank_vector
            return np.linalg.norm(perpendicular)
        
        far_corners = sorted(remaining_corners, key=perpendicular_distance, reverse=True)
        
        # Détermine a8 et h8 en fonction de leur position relative à a1 et h1
        h1_candidate = far_corners[0]
        h8_candidate = far_corners[1]
        
        # Si h8 est plus proche de h1 que de a1, on inverse
        if (np.linalg.norm(np.array(h8_candidate) - np.array(labeled_corners['a8'])) <
            np.linalg.norm(np.array(h1_candidate) - np.array(labeled_corners['a8']))):
            h1_candidate, h8_candidate = h8_candidate, h1_candidate
            
        labeled_corners['h8'] = tuple(h1_candidate)
        labeled_corners['h1'] = tuple(h8_candidate)

    return labeled_corners



def draw_stickers(img, blue_stickers, pink_stickers):
    """
    Dessine les autocollants bleus et roses sur l'image.
    """
    
    # Dessiner les stickers si présents
    if blue_stickers is not None and len(blue_stickers) > 0:
        cX, cY, radius = blue_stickers  # Déballage du tuple
        # radius = 5
        cv2.circle(img, (cX, cY), radius, (255, 0, 0), 2)  # Contour du sticker
        cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Centre du sticker

    if pink_stickers is not None and len(pink_stickers) > 0:
        cX, cY, radius = pink_stickers  # Déballage du tuple
        cv2.circle(img, (cX, cY), radius, (255, 0, 255), 2)  # Contour du sticker
        cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Centre du sticker


    # for (cX, cY, radius) in blue_stickers:
    #     cv2.circle(img, (cX, cY), radius, (255, 0, 0), 2)  # Blue outline
    #     cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Center in green

    # # Draw pink stickers
    # for (cX, cY, radius) in pink_stickers:
    #     cv2.circle(img, (cX, cY), radius, (255, 0, 255), 2)  # Pink outline
    #     cv2.circle(img, (cX, cY), 3, (0, 255, 0), -1)  # Center in green

    return img