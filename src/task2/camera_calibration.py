import numpy as np
import cv2
import glob
from tqdm import tqdm



def calibrate_camera_from_video(video_path, chessboard_size, frame_interval=100, show_process=False):
    """
    Calibre la caméra en utilisant des images extraites d'une vidéo à intervalles réguliers.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erreur lors de l'ouverture de la vidéo : {video_path}")
        return None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    pbar = tqdm(total=total_frames, desc="Calibration Progress")
    frame_count = 0


    while True:

        # Stop the loop if the end of the video is reached
        if frame_count >= total_frames:
            break
        
        if frame_count % frame_interval == 0:

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            frameFound, frame = cap.read()

            if not frameFound:
                break

            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cornersFound, corners = cv2.findChessboardCorners(imgGray, chessboard_size, 
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                     cv2.CALIB_CB_FAST_CHECK + 
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE)

            if cornersFound:
                objpoints.append(objp)
                cornersRefined = cv2.cornerSubPix(imgGray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(cornersRefined)

                if show_process:
                    # Optionnel : afficher les coins détectés
                    cv2.drawChessboardCorners(frame, chessboard_size, cornersRefined, cornersFound)
                    cv2.imshow('Chessboard Detection', frame)
                    cv2.waitKey(500)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    # print(f"Nombre d'images utilisées pour la calibration: {len(objpoints)}")

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) > 0:

        # Calibrate the camera
        print("Calibrating the camera...")
        repError, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error


        # Save the camera matrix and distortion coefficients to a file
        np.savez('camera_calibration_results.npz', cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)    
        
        print(f"Calibration effectuée avec {len(objpoints)} images")
        print(f"Erreur de reprojection moyenne: {mean_error/len(objpoints)}")

        return cameraMatrix, distCoeffs
    else:
        print("Aucun échiquier détecté dans la vidéo")
        return None, None
    

def undistort_frame(frame, camera_matrix, dist_coeffs):
    """
    Corrige la distorsion d'une image en utilisant les paramètres de calibration.
    """
    height, width = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width,height), 1, (width,height))
    
    # Undistort
    dst = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst



if __name__ == "__main__":
    # Paramètres de calibration
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    chessboard_size = (7, 7)
    frame_interval = 1000

    # calibrate_camera_from_video(video_path, chessboard_size, frame_interval=frame_interval, show_process=True)

    # Charger les résultats de la calibration
    calibration_results = np.load('camera_calibration_results.npz')
    camera_matrix = calibration_results['cameraMatrix']
    dist_coeffs = calibration_results['distCoeffs']

    if camera_matrix is not None and dist_coeffs is not None:
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:", dist_coeffs.ravel())

        # Test de la correction de distorsion sur une frame de la vidéo
        cap = cv2.VideoCapture(video_path)
        # ret, frame = cap.read()

        while True:

            # Lire la frame
            ret, frame = cap.read()
            if not ret:
                break

            undistorted_frame = undistort_frame(frame, camera_matrix, dist_coeffs)

            # Redimensionner l'image pour l'affichage
            display_width = 800  # Vous pouvez ajuster cette valeur selon vos besoins
            aspect_ratio = frame.shape[1] / frame.shape[0]
            display_height = int(display_width / aspect_ratio)

            undistorted_frame = cv2.resize(undistorted_frame, (display_width, display_height))
            frame = cv2.resize(frame, (display_width, display_height))

            # Afficher l'image traitée
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Undistorted Frame', undistorted_frame)

            # Attendre 1ms entre chaque frame et vérifier si l'utilisateur veut quitter
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or cv2.getWindowProperty('Original Frame', cv2.WND_PROP_VISIBLE) < 1 or cv2.getWindowProperty('Undistorted Frame', cv2.WND_PROP_VISIBLE) < 1:
                break

            # cv2.waitKey(0)
        cv2.destroyAllWindows()
        cap.release()
    else:
        print("La calibration a échoué.")




