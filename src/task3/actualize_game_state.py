import numpy as np

import task3.chessboard_utils as utils

# Définir le nombre de pièces de chaque type par couleur
PIECES_COUNT = {
    "pawn": 8,
    "rook": 2,
    "knight": 2,
    "bishop": 2,
    "queen": 1,
    "king": 1
}

# Au début du fichier, après PIECES_COUNT
IDENTIFIED_PIECES = {
    'white_pawn': 0, 'white_rook': 0, 'white_knight': 0,
    'white_bishop': 0, 'white_queen': 0, 'white_king': 0,
    'black_pawn': 0, 'black_rook': 0, 'black_knight': 0,
    'black_bishop': 0, 'black_queen': 0, 'black_king': 0
}



def get_piece_type(piece_name):
    """Extrait le type de pièce du nom complet (ex: 'white_pawn' -> 'pawn')"""
    return piece_name.split('_')[1]


def initialize_game_state(game_state):
    game_actualization = {
        'piece_certainty': {}
    }

    for i in range(0, 8):
        for j in range(0, 8):

            pos = (i, j)
            game_actualization['piece_certainty'][pos] = {}
            
            if game_state[i, j] > 0:
                game_actualization['piece_certainty'][pos] = {
                    'white_pawn': PIECES_COUNT['pawn']/16,
                    'white_rook': PIECES_COUNT['rook']/16,
                    'white_knight': PIECES_COUNT['knight']/16,
                    'white_bishop': PIECES_COUNT['bishop']/16,
                    'white_queen': PIECES_COUNT['queen']/16,
                    'white_king': PIECES_COUNT['king']/16
                }
            # Si c'est une pièce noire
            elif game_state[i, j] < 0:
                game_actualization['piece_certainty'][pos] = {
                    'black_pawn': PIECES_COUNT['pawn']/16,  
                    'black_rook': PIECES_COUNT['rook']/16,
                    'black_knight': PIECES_COUNT['knight']/16,
                    'black_bishop': PIECES_COUNT['bishop']/16,
                    'black_queen': PIECES_COUNT['queen']/16,
                    'black_king': PIECES_COUNT['king']/16
                }
            else:
                game_actualization['piece_certainty'][pos] = {}

    return game_actualization

def actualize_game_state(game_actualization, move_analysis, board):
    """
    Met à jour notre connaissance du jeu en fonction du mouvement analysé.
    
    Args:
        game_state (dict): État actuel de nos connaissances avec:
            - board (np.array): État du plateau
            - piece_certainty (dict): Dictionnaire de certitude pour chaque position
              format: {(row, col): {piece_type: probability}}
        move_analysis (dict): Résultat de l'analyse du mouvement avec:
            - valid (bool): si le mouvement est valide
            - from_pos (tuple): position de départ
            - to_pos (tuple): position d'arrivée
            - valid_pieces (list): liste des pièces possibles pour ce mouvement
        board (np.array): État actuel du plateau de jeu
    
    Returns:
        tuple: (game_state mis à jour, board mis à jour)
    """
    # Si le mouvement n'est pas valide, ne rien mettre à jour
    if not move_analysis['valid']:
        return game_actualization
    
    # Si c'est le premier appel, initialiser le dictionnaire de certitude
    if 'piece_certainty' not in game_actualization:
        game_actualization['piece_certainty'] = {}
    
    from_pos = move_analysis['from_pos']
    to_pos = move_analysis['to_pos']
    valid_pieces = move_analysis['valid_pieces']

    
    if from_pos not in game_actualization['piece_certainty']:
        game_actualization['piece_certainty'][from_pos] = {}
    
    if len(valid_pieces) > 0:
        # Calculer les poids en fonction du nombre de pièces de chaque type
        weights = []

        if len(valid_pieces) > 1:
            for piece in valid_pieces:
                piece_type = get_piece_type(piece)
                weights.append(PIECES_COUNT[piece_type] - IDENTIFIED_PIECES[piece])
        else:
            weights.append(1)
        
        # Normaliser les poids pour obtenir des probabilités
        total_weight = sum(weights)
        base_probabilities = [w / total_weight for w in weights]
        
        # Sauvegarder les probabilités de la position de départ
        # pour les utiliser dans la mise à jour
        last_probabilities = game_actualization['piece_certainty'][from_pos]

        new_probabilities = {}

        ###########################################
        # * PROBABILITIES UPDATE WITH WEIGHTS
        ###########################################
        
        for piece, base_prob in zip(valid_pieces, base_probabilities):
            if piece in last_probabilities:
                # Combiner l'historique avec les nouvelles probabilités pondérées
                current_prob = last_probabilities[piece]
                # On peut ajuster le poids relatif entre l'historique et les nouvelles infos
                history_weight = 0.7  # 70% historique, 30% nouvelle info
                new_prob = (current_prob * history_weight + 
                           base_prob * (1 - history_weight))
                new_probabilities[piece] = min(1.0, new_prob)
            elif last_probabilities == {}:
                # Première observation: utiliser directement les probabilités pondérées
                new_probabilities[piece] = base_prob
        
        
        
        ###########################################
        # * NORMALIZATION
        ###########################################
        
        # Normaliser les probabilités
        total_prob = sum(new_probabilities.values())
        if total_prob > 0:
            for piece in new_probabilities:
                new_probabilities[piece] /= total_prob

        # Mettre à jour la nouvelle position
        game_actualization['piece_certainty'][to_pos] = new_probabilities
        

        ###########################################
        # * UPDATE OTHER PIECES WHEN CERTAINTY REACHED
        ###########################################
        
        # Vérifier si une pièce atteint 100% de certitude
        for piece, prob in new_probabilities.items():
            if prob > 0.99:  # On utilise 0.99 pour gérer les erreurs d'arrondi
                # Vérifier si la pièce n'était pas déjà identifiée avec certitude
                was_certain = False
                if from_pos in game_actualization['piece_certainty']:
                    for old_piece, old_prob in game_actualization['piece_certainty'][from_pos].items():
                        if old_piece == piece and old_prob > 0.99:
                            was_certain = True
                            break
                
                # Incrémenter le compteur seulement si c'est une nouvelle identification
                if not was_certain:

                    print(f"NEW PIECE IDENTIFIED: {piece}")
                    IDENTIFIED_PIECES[piece] += 1
                    piece_type = get_piece_type(piece)
                    
                    # Si nous avons trouvé toutes les pièces de ce type
                    if IDENTIFIED_PIECES[piece] >= PIECES_COUNT[piece_type]:
                        # Parcourir toutes les positions pour supprimer cette possibilité
                        for pos, certainties in game_actualization['piece_certainty'].items():
                            if pos != to_pos and piece in certainties and certainties[piece] < 0.99:
                                del certainties[piece]
                                
                                # Renormaliser les probabilités restantes
                                if certainties:  # Si il reste des pièces possibles
                                    total_prob = sum(certainties.values())
                                    if total_prob > 0:
                                        for p in certainties:
                                            certainties[p] /= total_prob

        # Vider l'ancienne position
        game_actualization['piece_certainty'][from_pos] = {}
        

    # Clear void positions certainty
    for pos in game_actualization['piece_certainty']:
        if board[pos] == 0:
            game_actualization['piece_certainty'][pos] = {}
    
    # Mettre à jour le plateau si certitude suffisante
    CERTAINTY_THRESHOLD = 0.65
    for pos, certainties in game_actualization['piece_certainty'].items():
        for piece, prob in certainties.items():
            if prob > CERTAINTY_THRESHOLD:
                board[pos] = utils.PIECES_TO_NUM[piece]
    
    return game_actualization, board, new_probabilities




def actualize_game_state_with_castling(game_actualization, move_analysis, board):

    # Si le mouvement n'est pas valide, ne rien mettre à jour
    if not move_analysis['valid']:
        return game_actualization
    
    # Si c'est le premier appel, initialiser le dictionnaire de certitude
    if 'piece_certainty' not in game_actualization:
        game_actualization['piece_certainty'] = {}


    king_final_pos = move_analysis['king_final']
    rook_final_pos = move_analysis['rook_final']

    color = 'white' if move_analysis['color'] > 0 else 'black'

    game_actualization['piece_certainty'][king_final_pos] = {color + '_king': 1.0}
    game_actualization['piece_certainty'][rook_final_pos] = {color + '_rook': 1.0}

    new_probabilities = {
        'king': game_actualization['piece_certainty'][king_final_pos],
        'rook': game_actualization['piece_certainty'][rook_final_pos]
    }


    ###########################################
    # * UPDATE OTHER PIECES WHEN CERTAINTY REACHED
    ###########################################

    for pos, certainties in game_actualization['piece_certainty'].items():
        if pos != king_final_pos and color + '_king' in certainties:
            del certainties[color + '_king']

    for pos, certainties in game_actualization['piece_certainty'].items():

        if IDENTIFIED_PIECES[color + '_rook'] >= PIECES_COUNT['rook']:
            if pos != rook_final_pos and color + '_rook' in certainties and certainties[color + '_rook'] < 0.99:
                del certainties[color + '_rook']

    

    # Clear void positions certainty
    for pos in game_actualization['piece_certainty']:
        if board[pos] == 0:
            game_actualization['piece_certainty'][pos] = {}



    # Mettre à jour le plateau si certitude suffisante
    CERTAINTY_THRESHOLD = 0.6
    for pos, certainties in game_actualization['piece_certainty'].items():
        for piece, prob in certainties.items():
            if prob > CERTAINTY_THRESHOLD:
                board[pos] = utils.PIECES_TO_NUM[piece]
    
    return game_actualization, board, new_probabilities


    

