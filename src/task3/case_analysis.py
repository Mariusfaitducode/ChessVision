import numpy as np
import matplotlib.pyplot as plt
import cv2



def detect_if_case_is_occupied(edges_frame, inner_top, inner_left, inner_bottom, inner_right):
    
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




# def detect_piece_color(piece_region_bgr, is_dark_square, debug=True):
#     # Séparer les canaux BGR
#     b, g, r = cv2.split(piece_region_bgr)
    
#     def find_peaks(channel, min_peak_distance=30):
#         hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
#         hist_smooth = cv2.GaussianBlur(hist, (5, 1), 0)
        
#         peaks = []
#         min_peak_height = np.max(hist_smooth) * 0.01
        
#         for i in range(1, 255):
#             if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
#                 if hist_smooth[i] > min_peak_height:
#                     is_far_enough = True
#                     for existing_peak, _ in peaks:
#                         if abs(existing_peak - i) < min_peak_distance:
#                             is_far_enough = False
#                             if hist_smooth[i] > hist_smooth[existing_peak]:
#                                 peaks.remove((existing_peak, hist_smooth[existing_peak][0]))
#                                 is_far_enough = True
#                                 break
                    
#                     if is_far_enough:
#                         peaks.append((i, hist_smooth[i][0]))
        
#         return sorted(peaks, key=lambda x: x[1], reverse=True), hist_smooth
    
#     # Analyser chaque canal
#     peaks_b, hist_b = find_peaks(b)
#     peaks_g, hist_g = find_peaks(g)
#     peaks_r, hist_r = find_peaks(r)
    
#     if debug:
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
#         # Image originale
#         ax1.imshow(cv2.cvtColor(piece_region_bgr, cv2.COLOR_BGR2RGB))
#         ax1.set_title('Region analysée')
        
#         # Histogrammes des trois canaux
#         ax2.plot(hist_b, 'b-', label='Blue channel')
#         for pos, height in peaks_b:
#             ax2.plot(pos, height, 'bo', markersize=10)
#             ax2.text(pos, height, f' {pos}', verticalalignment='bottom')
#         ax2.grid(True)
#         ax2.legend()
#         ax2.set_title('Blue Channel Histogram')
        
#         ax3.plot(hist_g, 'g-', label='Green channel')
#         for pos, height in peaks_g:
#             ax3.plot(pos, height, 'go', markersize=10)
#             ax3.text(pos, height, f' {pos}', verticalalignment='bottom')
#         ax3.grid(True)
#         ax3.legend()
#         ax3.set_title('Green Channel Histogram')
        
#         ax4.plot(hist_r, 'r-', label='Red channel')
#         for pos, height in peaks_r:
#             ax4.plot(pos, height, 'ro', markersize=10)
#             ax4.text(pos, height, f' {pos}', verticalalignment='bottom')
#         ax4.grid(True)
#         ax4.legend()
#         ax4.set_title('Red Channel Histogram')
        
#         plt.tight_layout()
#         plt.show()
    
#     # Analyse des pics pour déterminer la couleur
#     def analyze_channel_peaks(peaks, channel_name):
#         if len(peaks) < 2:
#             return None
        
#         # Ignorer le pic du background et analyser les autres pics
#         for peak_pos, _ in peaks[1:]:
#             # Les valeurs exactes des seuils peuvent nécessiter des ajustements
#             if channel_name == 'red':
#                 if peak_pos > 180:  # Pièce blanche devrait avoir un pic élevé en rouge
#                     return 'white'
#                 elif peak_pos < 150:  # Pièce noire devrait avoir un pic bas en rouge
#                     return 'black'
#             elif channel_name == 'blue':
#                 if peak_pos > 180:
#                     return 'white'
#                 elif peak_pos < 150:
#                     return 'black'
#         return None
    
#     # Combiner les résultats des trois canaux
#     results = []
#     if len(peaks_r) > 1:
#         results.append(analyze_channel_peaks(peaks_r, 'red'))
#     if len(peaks_b) > 1:
#         results.append(analyze_channel_peaks(peaks_b, 'blue'))
#     if len(peaks_g) > 1:
#         results.append(analyze_channel_peaks(peaks_g, 'green'))
    
#     # Prendre une décision basée sur la majorité
#     results = [r for r in results if r is not None]
#     if not results:
#         return None
    
#     # Retourner la couleur la plus fréquente
#     return max(set(results), key=results.count)


def get_piece_roi(piece_region):
    # Appliquer Canny pour détecter les contours
    edges = cv2.Canny(piece_region, 50, 150)
    
    # Trouver les contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Trouver le plus grand contour (qui devrait être la pièce)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Obtenir le rectangle englobant
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Ajouter une petite marge autour de la pièce (20% de la taille)
    margin_x = int(w * 0.2)
    margin_y = int(h * 0.2)
    
    # Calculer les nouvelles coordonnées avec la marge
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(piece_region.shape[1], x + w + margin_x)
    y2 = min(piece_region.shape[0], y + h + margin_y)
    
    return (x1, y1, x2, y2)




def detect_piece_color(piece_region, is_dark_square, debug=False):

    # Prétraitement de l'image
    blurred = cv2.GaussianBlur(piece_region, (5, 5), 0)
    
    # Obtenir la ROI centrée sur la pièce
    roi_coords = get_piece_roi(blurred)
    if roi_coords is None:
        return None
    
    x1, y1, x2, y2 = roi_coords
    piece_roi = piece_region[y1:y2, x1:x2]


    # Calculer l'histogramme
    hist = cv2.calcHist([piece_roi], [0], None, [256], [0, 256])
    
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

    if debug:
        plt.figure(figsize=(15, 5))
        
        # Image originale avec ROI
        plt.subplot(131)
        plt.imshow(piece_region, cmap='gray')
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
        plt.title('Image originale avec ROI')
        
        # ROI extraite
        plt.subplot(132)
        plt.imshow(piece_roi, cmap='gray')
        plt.title('ROI de la pièce')
        
        # Histogramme
        plt.subplot(133)
        plt.plot(hist_smooth, 'b-', label='Histogramme lissé')
        plt.plot(hist, 'r:', alpha=0.5, label='Histogramme brut')
        
        for i, (pos, height) in enumerate(peaks):
            plt.plot(pos, height, 'go', markersize=10, label=f'Pic {i+1} (val={pos})')
            plt.text(pos, height, f'  {pos}', verticalalignment='bottom')
        
        plt.title(f"Histogramme de la ROI")
        plt.legend()
        plt.grid(True)
        plt.show()
    


    
    if len(peaks) < 2:
        return None, None  # Pas assez de pics pour une détection fiable
    
    # Le plus grand pic correspond au fond
    background_peak = peaks[0][0]
    # Le deuxième pic le plus important correspond à la pièce
    piece_peak = peaks[1][0]
            
    if abs(piece_peak - 130) < 40:
        return 'white', piece_peak
    
    # Pièce blanche (pic autour de 210)
    elif abs(piece_peak - 210) < 30:
        return 'black', piece_peak
    
    return None, piece_peak
