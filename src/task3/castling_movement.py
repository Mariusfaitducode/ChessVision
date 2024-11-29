

#  * Verify if the king was at the good position

def verify_king_step(positions, prev_state):

    # King positons must be E1 or E8
    king_positions = [(0, 4), (7, 4)]

    row = None
    color = None

    # Verify if the white king changed and was in the good position
    if king_positions[0] in positions and prev_state[king_positions[0]] > 0:
        # White king
        row = 0
        color = 1

    # Verify if the black king changed was in the good position
    elif king_positions[1] in positions and prev_state[king_positions[1]] < 0:
        # Black king
        row = 7
        color = -1
    else:
        return False, None, None

    return True, row, color


# * Verify if the rook was in the good position and with the good color

def verify_rook_step(positions, prev_state, curr_state, row, color):

    rook_positions = [(row, 0), (row, 7)]

    side = None

    # Verify if the left rook changed was in the good position and with the right color
    if rook_positions[0] in positions and prev_state[rook_positions[0]]*color > 0:
        
        # Left castling Queenside
        king_final = (row, 2)
        rook_final = (row, 3)
        side = 'queenside'

    # Verify if the right rook changed was in the good position and with the right color
    elif rook_positions[1] in positions and prev_state[rook_positions[1]]*color > 0:

        # Right castling Kingside
        king_final = (row, 6)
        rook_final = (row, 5)
        side = 'kingside'
    else:
        return False, None, None, None
    
    return True, king_final, rook_final, side

            

def verify_final_positions(king_final, rook_final, prev_state, curr_state, color):
    # Verify if the castling move was valid
    if prev_state[rook_final] == 0 and prev_state[king_final] == 0:

        # Verify if the castling move was well done
        if curr_state[king_final]*color > 0 and curr_state[rook_final]*color > 0:

            return True
    return False



def castling_movement(positions, prev_state, curr_state):

    # ! Array of game in bad format -> A1 = H1 -> H = 0 and E = 3
    # Apply vertical symmetry to correct board orientation (A1 should be (0,0) not (0,7))
    # prev_state = np.fliplr(prev_state.copy())
    # curr_state = np.fliplr(curr_state.copy())
    
    # Apply same transformation to positions array
    # positions = [(pos[0], 7-pos[1]) for pos in positions]

    print('Castling positions', positions)

    valid, row, color = verify_king_step(positions, prev_state)

    if valid is False:
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'King position not found'
        }
    

    valid, king_final, rook_final, side = verify_rook_step(positions, prev_state, curr_state, row, color)
    
    if valid is False:
        return{
            'valid': False,
            'move_type': 'invalid',
            'message': 'Rook position not found'
        }
    

    valid = verify_final_positions(king_final, rook_final, prev_state, curr_state, color)

    # initial_positions = [pos for pos in positions if curr_state[pos] == 0]

    if valid is True:
        return {
                'valid': True,
                'move_type': 'castling',
                'message': f'{side} castling move is valid',
                'king_final': king_final,
                'rook_final': rook_final,
                'color': color
            }

    return {
        'valid': False,
        'move_type': 'invalid',
        'message': 'Castling move is invalid'
    }


def detect_first_part_castling(initial_pos, final_pos, prev_state):
    """
    Détecte si le mouvement du roi pourrait être la première partie d'un castling
    """
    # King positions must be E1 or E8
    king_positions = [(0, 4), (7, 4)]
    
    # Vérifier si c'est un roi qui bouge depuis sa position initiale
    if initial_pos not in king_positions:
        return None
        
    color = prev_state[initial_pos]
    row = initial_pos[0]
        
    # Détecter le type de castling potentiel
    if final_pos == (row, 6):  # Petit roque (kingside)
        return {
            'type': 'potential_castling',
            'side': 'kingside',
            'row': row,
            'color': color,
            'king_move': {'from': initial_pos, 'to': final_pos},
            'expected_rook': {
                'initial': (row, 7),
                'final': (row, 5)
            }
        }
    elif final_pos == (row, 2):  # Grand roque (queenside)
        return {
            'type': 'potential_castling',
            'side': 'queenside',
            'row': row,
            'color': color,
            'king_move': {'from': initial_pos, 'to': final_pos},
            'expected_rook': {
                'initial': (row, 0),
                'final': (row, 3)
            }
        }
        
    return None


def verify_second_part_castling(potential_castling, initial_pos, final_pos, prev_state):
    """
    Vérifie si le mouvement actuel pourrait être la seconde partie d'un castling
    
    Args:
        potential_castling: Dict contenant les informations du potentiel castling précédent
        initial_pos: Position initiale de la pièce qui bouge
        prev_state: État du plateau avant le mouvement
    
    Returns:
        dict: Information sur la position finale attendue de la tour si c'est un castling valide
        None: Si ce n'est pas un castling valide
    """
        
    # Vérifier que la tour est de la même couleur que le roi
    if (prev_state[initial_pos] > 0) != (potential_castling['color'] > 0):
        return None
        
    row = potential_castling['row']
    color = potential_castling['color']
    king_final = potential_castling['king_move']['to']
    
    # Vérifier la position initiale de la tour selon le type de castling
    if potential_castling['side'] == 'kingside':
        if initial_pos != (row, 7):  # Tour doit être en H1/H8
            return None
        
        if final_pos != (row, 5):
            return None

        return {
                'valid': True,
                'move_type': 'castling',
                'message': f'kingside castling move is valid',
                'king_final': king_final,
                'rook_final': final_pos,
                'color': color
            }
    else:  # queenside
        if initial_pos != (row, 0):  # Tour doit être en A1/A8
            return None

        if final_pos != (row, 3):
            return None
        
        return {
                'valid': True,
                'move_type': 'castling',
                'message': f'queenside castling move is valid',
                'king_final': king_final,
                'rook_final': final_pos,
                'color': color
            }
    
    return None

    