import numpy as np

import chessboard_utils as utils

def actualize_game_state(game_state, move_analysis, board):
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
    
    Returns:
        dict: État mis à jour de nos connaissances
    """
    if not move_analysis['valid']:
        return game_state
    
    # Si c'est le premier appel, initialiser le dictionnaire de certitude
    if 'piece_certainty' not in game_state:
        game_state['piece_certainty'] = {}
    
    from_pos = move_analysis['from_pos']
    to_pos = move_analysis['to_pos']
    valid_pieces = move_analysis['valid_pieces']
    
    # Mettre à jour les probabilités pour la position d'arrivée
    if to_pos not in game_state['piece_certainty']:
        game_state['piece_certainty'][to_pos] = {}
    
    # Nombre de pièces valides pour ce mouvement
    n_valid_pieces = len(valid_pieces)
    
    if n_valid_pieces > 0:
        # Distribution uniforme de probabilité entre les pièces valides
        prob_per_piece = 1.0 / n_valid_pieces
        
        # Mettre à jour les probabilités
        for piece in valid_pieces:
            if piece in game_state['piece_certainty'][to_pos]:
                # Augmenter la probabilité si la pièce était déjà considérée
                current_prob = game_state['piece_certainty'][to_pos][piece]
                game_state['piece_certainty'][to_pos][piece] = min(1.0, current_prob + prob_per_piece)
            else:
                # Initialiser la probabilité pour une nouvelle pièce
                game_state['piece_certainty'][to_pos][piece] = prob_per_piece
        
        # Normaliser les probabilités
        total_prob = sum(game_state['piece_certainty'][to_pos].values())
        if total_prob > 0:
            for piece in game_state['piece_certainty'][to_pos]:
                game_state['piece_certainty'][to_pos][piece] /= total_prob
    
    # Si une pièce atteint une certitude suffisante (ex: >0.9)
    # nous pouvons la considérer comme identifiée
    CERTAINTY_THRESHOLD = 0.9
    for pos, certainties in game_state['piece_certainty'].items():
        for piece, prob in certainties.items():
            if prob > CERTAINTY_THRESHOLD:
                # game_state['board'][pos] = piece

                board[pos] = utils.PIECES_TO_NUM[piece]
    
    return game_state, board




