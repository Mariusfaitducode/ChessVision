import cv2
import numpy as np


sticker_history = {'blue': None, 'pink': None}
labeled_corners = {'a1': None, 'a8': None, 'h8': None, 'h1': None}


def detect_stickers(img, distance_threshold=150):
    """
    Détecte les autocollants bleus et roses sur l'image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

    # range for blue sticker
    # lower_blue = np.array([90, 98, 182])  # Lower bound in HSV
    # upper_blue = np.array([100, 198, 248])
    # range for blue sticker
    lower_blue = np.array([90, 150, 100])  # Lower bound in HSV
    upper_blue = np.array([110, 255, 200])

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
        # Find a1 (closest to blue sticker)
        cX, cY, radius = blue_sticker
        sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
        labeled_corners['a1'] = tuple(sorted_corners[0])

    if pink_sticker is not None:
        # Find h1 (closest to pink sticker)
        cX, cY, radius = pink_sticker
        sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
        labeled_corners['a8'] = tuple(sorted_corners[0])



    if labeled_corners['a1'] is not None and labeled_corners['a8'] is not None:
        # Calculate reference vector (a1 to a8)
        reference_vector = np.array(labeled_corners['a8']) - np.array(labeled_corners['a1'])
        
        # Find the two remaining corners
        remaining_corners = [corner for corner in corners 
                           if corner != labeled_corners['a1'] and corner != labeled_corners['a8']]
        
        # Sort the remaining corners by the angle formed with the reference vector
        def angle_with_reference(point):
            point_vector = np.array(point) - np.array(labeled_corners['a1'])
            # Calculate the angle between the vectors in radians
            cos_angle = np.dot(point_vector, reference_vector) / (np.linalg.norm(point_vector) * np.linalg.norm(reference_vector))
            # Clip to avoid rounding errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            return angle
        
        angle_corners = sorted(remaining_corners, key=angle_with_reference, reverse=True)
        
        # Determine a8 and h8 based on their relative position to a1 and h1
        h1_candidate = angle_corners[1]
        h8_candidate = angle_corners[0]
        
        # If h8 is closer to h1 than to a1, invert them
            
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

    return img