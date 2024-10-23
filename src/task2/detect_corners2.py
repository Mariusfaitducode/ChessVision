import cv2
import numpy as np

def detect_chessboard(img, show_process=False, last_area=None):
    """
    Détecte l'échiquier dans l'image et retourne ses coins.
    
    :param img: Image d'entrée en couleur (BGR)
    :param show_process: Booléen pour afficher les étapes intermédiaires
    :return: Liste des coordonnées des coins [a1, a8, h8, h1] et contour approché, ou (None, None) si non détecté
    """
    # Conversion de l'image en niveaux de gris pour simplifier le traitement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # if show_process:
    #     cv2.imshow('Gray Image', gray)
    #     cv2.waitKey(100)

    # Appliquer un flou gaussien pour réduire le bruit
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Détection des contours avec l'algorithme de Canny
    # Les seuils 50 et 150 sont utilisés pour détecter les contours faibles et forts
    edges = cv2.Canny(gray, 50, 150)
    
    # if show_process:
    #     cv2.imshow('Canny Edges', edges)
    #     cv2.waitKey(100)

    # Trouver les contours dans l'image des bords
    # RETR_TREE récupère tous les contours et crée une hiérarchie complète
    # CHAIN_APPROX_SIMPLE compresse les segments horizontaux, verticaux et diagonaux
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Trier les contours par aire décroissante
    # L'hypothèse est que l'échiquier sera l'un des plus grands contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    img_area = img.shape[0] * img.shape[1]
    min_board_area = img_area * 0.1  # Le plateau doit occuper au moins 20% de l'image
    max_board_area = img_area * 0.9  # Le plateau ne doit pas occuper plus de 90% de l'image


    for contour in contours:
        # Approximation du contour pour réduire le nombre de points
        # epsilon est la précision de l'approximation, plus il est petit, plus l'approximation est précise
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if show_process:
            # Dessiner le contour approximé sur l'image originale
            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            cv2.imshow('Contours', img)
            cv2.waitKey(100)

        # Si le contour a 4 côtés, c'est potentiellement notre échiquier
        if len(approx) == 4:

            # * Vérifier si le contour est proche d'un rectangle

            # if not is_approximately_rectangular(approx):
            #     print("Not a rectangle")
            #     continue

            # *Vérifier l'aire du contour

            area = cv2.contourArea(contour)  # Utiliser le contour original au lieu de l'approximation

            # Vérifier si l'aire est dans les 10% de la dernière aire détectée
            if last_area is not None and abs(area - last_area) > 0.5 * last_area:
                # print(f"Area {area} is not within 10% of last area {last_area}")
                continue


            


            

            

            

            # elif area < min_board_area or area > max_board_area:
            #     print(f"Area {area} is not between min {min_board_area} and max {max_board_area}")
            #     continue

            # print(f"Area {area} is between min {min_board_area} and max {max_board_area}")

            # # *Vérifier le ratio d'aspect

            # print("Checking aspect ratio")

            # rect = cv2.minAreaRect(approx)
            # width, height = rect[1]

            # print(f"Width {width}, height {height}")
            # aspect_ratio = min(width, height) / max(width, height)
            # if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # Le plateau d'échecs devrait être presque carré
            #     print("Aspect ratio is not between 0.7 and 1.3")
            #     continue



            # Réorganiser les points du contour
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            (tl, tr, br, bl) = rect


            # Vérifier s'il n'y a pas de points superposés
            unique_points = np.unique(rect, axis=0)
            if len(unique_points) < 4:
                # print("Points superposés détectés, contour ignoré")
                continue


            # Calculer la largeur du rectangle transformé
            widthA = np.linalg.norm(br - bl)
            widthB = np.linalg.norm(tr - tl)
            maxWidth = max(int(widthA), int(widthB))

            # Calculer la hauteur du rectangle transformé
            heightA = np.linalg.norm(tr - br)
            heightB = np.linalg.norm(tl - bl)
            maxHeight = max(int(heightA), int(heightB))

            # Définir les points de destination pour la transformation de perspective
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            # Calculer la matrice de transformation de perspective
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Appliquer la transformation de perspective
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            
            if show_process:
                cv2.imshow('Warped Image', warped)
                cv2.waitKey(100)

            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Calculer la taille d'une case de l'échiquier
            square_size = maxWidth // 8  # en supposant un échiquier 8x8

            # # Définir les coordonnées des coins de l'échiquier dans l'image transformée
            # a1_target = np.array([square_size * 0, square_size * 7])  # Bottom-left (a1)
            # a8_target = np.array([square_size * 0, square_size * 0])  # Top-left (a8)
            # h1_target = np.array([square_size * 7, square_size * 7])  # Bottom-right (h1)
            # h8_target = np.array([square_size * 7, square_size * 0])  # Top-right (h8)

            # Calculer la matrice de transformation inverse
            # reverse_M = cv2.getPerspectiveTransform(dst, rect)

            # Fonction pour appliquer la transformation inverse à un point
            # def warp_point(point, M):
            #     point = np.array([point], dtype='float32')
            #     point = np.array([point])
            #     return cv2.perspectiveTransform(point, M)[0][0]

            # # Appliquer la transformation inverse pour obtenir les coordonnées des coins dans l'image originale
            # a1_orig = warp_point(a1_target, reverse_M)
            # a8_orig = warp_point(a8_target, reverse_M)
            # h1_orig = warp_point(h1_target, reverse_M)
            # h8_orig = warp_point(h8_target, reverse_M)

            # Réorganiser les coins dans l'ordre [a1, a8, h8, h1], cette ordre dépend de la position du rectangle dans l'image
           
            (h8, h1, a1, a8) = rect

            # print(f"a1: {a1}, a8: {a8}, h8: {h8}, h1: {h1}")

            # Retourner les coins dans l'ordre [a1, a8, h8, h1] et le contour approché
            return [a1, a8, h8, h1], approx, area

    # Si aucun échiquier n'est détecté, retourner None, None
    return None, None, last_area



def is_approximately_rectangular(points):
    """
    Vérifie si les points forment approximativement un rectangle.
    """
    if len(points) != 4:
        return False

    points = points.reshape(4, 2)
    edges = np.roll(points, 1, axis=0) - points
    lengths = np.sqrt((edges ** 2).sum(axis=1))
    angles = np.abs(np.degrees(np.arctan2(edges[:, 1], edges[:, 0])))
    
    # Vérifier si les côtés opposés ont des longueurs similaires
    if not (0.8 < lengths[0] / lengths[2] < 1.2 and 0.8 < lengths[1] / lengths[3] < 1.2):
        return False
    
    # Vérifier si les angles sont proches de 90 degrés
    if not all(80 < angle < 100 or 260 < angle < 280 for angle in angles):
        return False

    return True


def order_points(pts):
    """
    Ordonne les points dans l'ordre : haut-gauche, haut-droit, bas-droit, bas-gauche
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect



def draw_chessboard(img, corners, approx):
    """
    Dessine l'échiquier détecté sur l'image en utilisant polylines.
    """
    if corners is None or approx is None:
        return img

    # cv2.polylines(img, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

    if corners is not None:
        # Convertir les coins en un tableau numpy pour polylines
        corners_array = np.array(corners, dtype=np.int32)
        
        # Dessiner le contour de l'échiquier
        cv2.polylines(img, [corners_array], True, (0, 255, 0), 2)
        
        # Dessiner les coins individuels avec leurs labels
        corner_labels = ['a1', 'a8', 'h8', 'h1']
        for i, corner in enumerate(corners):
            corner_int = tuple(corner.astype(int))
            cv2.circle(img, corner_int, 5, (0, 0, 255), -1)
            cv2.putText(img, corner_labels[i], (corner_int[0] + 10, corner_int[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img



# Cette partie ne s'exécutera que si le script est exécuté directement (pas importé)
if __name__ == "__main__":
    # Tester la fonction avec une seule image
    img = cv2.imread('src/calibration_images/img1.png')

    # Charger une vidéo de test
    video_path = 'videos/moving_game.MOV'  # Remplacez par le chemin de votre vidéo
    cap = cv2.VideoCapture(video_path)

    # ret, frame0 = cap.read()
    

    # corners, approx = detect_chessboard(frame0, show_process=True)

    # if corners is not None:
    #     img_with_chessboard = draw_chessboard(frame0.copy(), corners, approx)
    #     cv2.imshow("Detected Chessboard", img_with_chessboard)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    


    frame_interval = 10
    frame_count = 0

    last_area = None

    

    if not cap.isOpened():
        print(f"Erreur: Impossible de charger la vidéo {video_path}")
    else:
        while True:

            if frame_count % frame_interval == 0:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()

                if not ret:
                    break

                

                corners, approx, last_area = detect_chessboard(frame, show_process=False, last_area=last_area)
    
                if corners is not None:
                    img_with_chessboard = draw_chessboard(frame.copy(), corners, approx)


                    display_width = 800  # Vous pouvez ajuster cette valeur selon vos besoins
                    aspect_ratio = img_with_chessboard.shape[1] / img_with_chessboard.shape[0]
                    display_height = int(display_width / aspect_ratio)
                    display_frame = cv2.resize(img_with_chessboard, (display_width, display_height))

                    # Afficher l'image traitée
                    # cv2.imshow('Processed Video', display_frame)

                    cv2.imshow("Detected Chessboard", display_frame)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                else:
                    print("No chessboard detected")


                key = cv2.waitKey(100) & 0xFF

                if key == ord('q'):
                    print("Fermeture des fenêtres")
                    break

            frame_count += 1

        cap.release()
