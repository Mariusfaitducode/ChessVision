import cv2
import numpy as np

img = cv2.imread('img1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:
        cv2.polylines(img, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

        pts = approx.reshape(4, 2)

        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        # Now detect the exact four corners of the chessboard squares (a1, a8, h1, h8)
        square_size = maxWidth // 8  # assuming an 8x8 board

        a1_target = np.array([square_size * 0, square_size * 7])  # Bottom-left (a1)
        a8_target = np.array([square_size * 0, square_size * 0])  # Top-left (a8)
        h1_target = np.array([square_size * 7, square_size * 7])  # Bottom-right (h1)
        h8_target = np.array([square_size * 7, square_size * 0])  # Top-right (h8)

        # Draw circles for each target corner on the warped image
        cv2.circle(warped, tuple(a1_target), 5, (0, 255, 0), -1)  # Green circle for a1
        cv2.circle(warped, tuple(a8_target), 5, (0, 255, 0), -1)  # Green circle for a8
        cv2.circle(warped, tuple(h1_target), 5, (0, 255, 0), -1)  # Green circle for h1
        cv2.circle(warped, tuple(h8_target), 5, (0, 255, 0), -1)  # Green circle for h8

        # Warp the corner points back to the original image
        reverse_M = cv2.getPerspectiveTransform(dst, rect)

        # Apply the inverse perspective transformation to the target points
        def warp_point(point, M):
            point = np.array([point], dtype='float32')
            point = np.array([point])
            return cv2.perspectiveTransform(point, M)[0][0]

        # Warp back to the original image for each point
        a1_orig = warp_point(a1_target, reverse_M)
        a8_orig = warp_point(a8_target, reverse_M)
        h1_orig = warp_point(h1_target, reverse_M)
        h8_orig = warp_point(h8_target, reverse_M)

        # Draw the four chessboard corners back on the original image
        cv2.circle(img, tuple(a1_orig.astype(int)), 5, (255, 0, 0), -1)  # Blue circle for a1
        cv2.circle(img, tuple(a8_orig.astype(int)), 5, (255, 0, 0), -1)  # Blue circle for a8
        cv2.circle(img, tuple(h1_orig.astype(int)), 5, (255, 0, 0), -1)  # Blue circle for h1
        cv2.circle(img, tuple(h8_orig.astype(int)), 5, (255, 0, 0), -1)  # Blue circle for h8

        # Show the final image with both the chessboard polygon and the square corners marked
        cv2.imshow("Original Image with Chessboard and Square Corners", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        break  # Exit after finding the first quadrilateral
