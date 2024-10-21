import cv2
import numpy as np

def detect_stickers(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

    # range for blue sticker
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # range for pink sticker
    lower_pink = np.array([140, 100, 100])
    upper_pink = np.array([170, 255, 255])

    # threshold to get only blue and pink colors
    mask_blue = cv2.inRange(blurred, lower_blue, upper_blue)
    mask_pink = cv2.inRange(blurred, lower_pink, upper_pink)

    blue_stickers = []
    pink_stickers = []

    # find contours for blue sticker
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)
    if contours_blue:
        for c in contours_blue:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            blue_stickers.append((int(cX), int(cY), int(radius)))

    # find contours for pink sticker
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink = sorted(contours_pink, key=cv2.contourArea, reverse=True)
    if contours_pink:
        for c in contours_pink:
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            pink_stickers.append((int(cX), int(cY), int(radius)))

    return blue_stickers, pink_stickers

def draw_stickers(img, blue_stickers, pink_stickers):
    """
    Dessine les autocollants détectés sur l'image.
    """
    img_copy = img.copy()
    
    # Dessiner les autocollants bleus
    for (cX, cY, radius) in blue_stickers:
        cv2.circle(img_copy, (cX, cY), radius, (255, 0, 0), 2)  # Contour bleu
        cv2.circle(img_copy, (cX, cY), 3, (0, 255, 0), -1)  # Centre vert
    
    # Dessiner les autocollants roses
    for (cX, cY, radius) in pink_stickers:
        cv2.circle(img_copy, (cX, cY), radius, (255, 0, 255), 2)  # Contour rose
        cv2.circle(img_copy, (cX, cY), 3, (0, 255, 0), -1)  # Centre vert
    
    return img_copy

# Cette partie ne s'exécutera que si le script est exécuté directement (pas importé)
if __name__ == "__main__":
    # Test the function with a single image
    img = cv2.imread('src/calibration_images/img1.png')
    blue_stickers, pink_stickers = detect_stickers(img)
    
    print("Blue stickers:", blue_stickers)
    print("Pink stickers:", pink_stickers)
    
    # Dessiner les autocollants pour la visualisation
    img_with_stickers = draw_stickers(img, blue_stickers, pink_stickers)
    
    cv2.imshow("Detected Stickers", img_with_stickers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
