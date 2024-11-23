import numpy as np
import matplotlib.pyplot as plt
import cv2



def detect_if_case_is_occupied(edges_frame, inner_top, inner_left, inner_bottom, inner_right):


    # Calculate margins in pixels
    # margin_h = int(square_h * margin_percent)
    # margin_w = int(square_w * margin_percent)
    
    # # Analysis zone coordinates (with margins)
    # inner_top = top + margin_h
    # inner_left = left + margin_w
    # inner_bottom = bottom - margin_h
    # inner_right = right - margin_w
    
    # Extract square region for edge detection (inner area only)
    square_edges = edges_frame[inner_top:inner_bottom, inner_left:inner_right]
    
    # Count edge pixels in inner area
    edge_count = np.count_nonzero(square_edges)
    
    # Calculate edge pixel percentage
    # Note: using inner area for calculation
    inner_area = (inner_bottom - inner_top) * (inner_right - inner_left)
    edge_percentage = (edge_count / inner_area) * 100
    
    # Threshold to determine if square is occupied
    edge_threshold = 1  # Adjust this threshold as needed
    
    # A square is considered occupied if it contains enough edges
    is_occupied = edge_percentage > edge_threshold

    return is_occupied, edge_percentage



def detect_piece_color(piece_region, is_dark_square):
    # Calculer l'histogramme
    hist = cv2.calcHist([piece_region], [0], None, [256], [0, 256])
    
    # Lisser l'histogramme pour réduire le bruit
    hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)

    # Trouver les pics significatifs
    peaks = []
    min_peak_height = np.max(hist_smooth) * 0.01  # 1% du maximum pour détecter les petits pics

    if is_dark_square:
        min_peak_distance = 15  # Distance minimale entre deux pics
    else:
        min_peak_distance = 8  # Distance minimale entre deux pics
    
    # Trouver les pics significatifs
    peaks = []
    for i in range(1, 255):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            # Vérifier si le pic est assez haut
            if hist_smooth[i] > min_peak_height:
                # Vérifier si le pic est assez loin des pics existants
                is_far_enough = True
                for existing_peak, _ in peaks:
                    if abs(existing_peak - i) < min_peak_distance:
                        is_far_enough = False
                        # Si le nouveau pic est plus haut, remplacer l'ancien
                        if hist_smooth[i] > hist_smooth[existing_peak]:
                            peaks.remove((existing_peak, hist_smooth[existing_peak][0]))
                            is_far_enough = True
                            break
                
                if is_far_enough:
                    peaks.append((i, hist_smooth[i][0]))
    
    # Trier les pics par hauteur
    peaks.sort(key=lambda x: x[1], reverse=True)


    
    # plt.figure(figsize=(12, 6))
    
    # # Subplot pour l'image de la case
    # plt.subplot(121)
    # plt.imshow(piece_region, cmap='gray')
    # plt.title('Region analysée')
    
    # # Subplot pour l'histogramme
    # plt.subplot(122)
    # plt.plot(hist_smooth, 'b-', label='Histogramme lissé')
    # plt.plot(hist, 'r:', alpha=0.5, label='Histogramme brut')
    
    # # Marquer les pics trouvés
    # for i, (pos, height) in enumerate(peaks):
    #     plt.plot(pos, height, 'go', markersize=10, label=f'Pic {i+1} (val={pos})')
    #     plt.text(pos, height, f'  {pos}', verticalalignment='bottom')
    
    # plt.title(f"Histogramme (Case {'sombre' if is_dark_square else 'claire'})")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    
    if len(peaks) < 2:
        return None, None  # Pas assez de pics pour une détection fiable
    
    # Le plus grand pic correspond au fond
    background_peak = peaks[0][0]
    # Le deuxième pic le plus important correspond à la pièce
    piece_peak = peaks[1][0]
    
    # Vérifier si les valeurs correspondent aux critères attendus
    # if is_dark_square:
    #     # Case noire (fond autour de 100)
    #     if abs(background_peak - 100) < 30:
    #         # Pièce noire (pic autour de 150)
    #         if abs(piece_peak - 150) < 20:
    #             return 'black'
    #         # Pièce blanche (pic autour de 210)
    #         elif abs(piece_peak - 210) < 20:
    #             return 'white'
    # else:
    #     # Case blanche (fond autour de 225)
    #     if abs(background_peak - 225) < 30:
    #         # Pièce noire (pic autour de 150)
    #         if abs(piece_peak - 150) < 20:
    #             return 'black'
    #         # Pièce blanche (pic autour de 210)
    #         elif abs(piece_peak - 210) < 20:
    #             return 'white'
            
    if abs(piece_peak - 130) < 40:
        return 'white', piece_peak
    
    # Pièce blanche (pic autour de 210)
    elif abs(piece_peak - 210) < 30:
        return 'black', piece_peak
    
    return None, piece_peak
