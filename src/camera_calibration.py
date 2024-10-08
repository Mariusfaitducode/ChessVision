
import numpy as np
import cv2
import glob



# This code won't run if this file is imported.

# Dimensions de l’échiquier utilisé (nombre de coins internes)
chessboard_size = (7, 7)
frame_size = (640, 480)


# Préparation des critères d’arrêt de l’algorithme
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Coordonnées 3D des points d'intersection de l’échiquier dans le monde réel
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)


# Listes pour stocker les points 3D du monde réel et les points 2D des images
objpoints = []
imgpoints = []


# Lire toutes les images d'un dossier
images = glob.glob('src/calibration_images/*.png')

print("images : ", images)


for fname in images:

    # Lire l'image
    img = cv2.imread(fname)
    # cv2.imshow('img', img)
    # cv2.waitKey(500)
    
    # # On convertit en gris pour la détection des coins
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray img', gray)
    # cv2.waitKey(500)
    
    # # Détection des coins de l'échiquier
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    print(ret, corners)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Afficher les coins détectés
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)

        cv2.imshow('img', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()


# Calibration de la caméra
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# ret is a boolean value indicating if the calibration was successful or not
# mtx is the camera matrix containing the intrinsic parameters of the camera
# dist is a vector containing the distortion coefficients of the camera
# rvecs is a list of rotation vectors for each calibration image
# tvecs is a list of translation vectors for each calibration image

print("Camera calibrated: ", ret)
print("Camera matrix: ", mtx)
print("Distortion coefficients: ", dist)
print("Rotation vectors: ", rvecs)
print("Translation vectors: ", tvecs)
