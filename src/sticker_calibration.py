import cv2
import numpy as np
from skimage import measure

img = cv2.imread('img1.png')
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

labels = measure.label(mask_pink)

# find contours for blue sticker
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours_blue = imutils.grab_contours(contours_blue)
contours_blue = sorted(contours_blue, key=cv2.contourArea, reverse=True)
if contours_blue:
    for (i, c) in enumerate(contours_blue):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 3)
        print(f'Coordinates for Blue Sticker {int(cX)}, {int(cY)} corresponding to the white side')

contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours_pink = imutils.grab_contours(contours_pink)
contours_pink = sorted(contours_pink, key=cv2.contourArea, reverse=True)
if contours_pink:
    for (i, c) in enumerate(contours_pink):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(img, (int(cX), int(cY)), int(radius),
                   (0, 0, 255), 3)
        print(f'Coordinates for Pink Sticker {int(cX)}, {int(cY)} corresponding to the black side')

cv2.imshow("Detected Stickers", img)
cv2.waitKey(0)
