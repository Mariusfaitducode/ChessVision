import cv2
import numpy as np



def detect_stickers(img, corners, distance_threshold=100):
    """
    Détecte les autocollants bleus et roses sur l'image.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (15, 15), 0)

    # range for blue sticker
    lower_blue = np.array([90, 80, 200])
    upper_blue = np.array([130, 255, 255])
    # lower_blue = np.array([90, 100, 50])
    # upper_blue = np.array([150, 205, 255])

    # range for pink sticker
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])

    # threshold to get only blue and pink colors
    mask_blue = cv2.inRange(blurred, lower_blue, upper_blue)
    kernel = np.ones((5, 5), np.uint8)
    mask_blue = cv2.erode(mask_blue, kernel, iterations=1)
    mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)

    mask_pink = cv2.inRange(blurred, lower_pink, upper_pink)

    blue_stickers = None
    pink_stickers = None

    labeled_corners = {'a1': None, 'a8': None, 'h8': None, 'h1': None}

    # Detect blue stickers
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)
    if contours_blue:
        for c in contours_blue:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
            blue_stickers = (int(cX), int(cY), int(radius))

            labeled_corners['a1'] = tuple(sorted_corners[0])
            labeled_corners['h1'] = tuple(sorted_corners[1])
            break

    # Detect pink stickers
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink = sorted(contours_pink, key=cv2.contourArea, reverse=True)
    if contours_pink:
        for c in contours_pink:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            sorted_corners = sorted(corners, key=lambda corner: np.linalg.norm(np.array([cX, cY]) - np.array(corner)))
            pink_stickers = (int(cX), int(cY), int(radius))

            labeled_corners['a8'] = tuple(sorted_corners[0])
            labeled_corners['h8'] = tuple(sorted_corners[1])
            break

    return blue_stickers, pink_stickers, labeled_corners



def draw_stickers(img, blue_stickers, pink_stickers):
    """
    Dessine les autocollants bleus et roses sur l'image.
    """
    
    # Dessiner les stickers si présents
    if blue_stickers is not None and len(blue_stickers) > 0:
        cX, cY, radius = blue_stickers  # Déballage du tuple
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