import numpy as np

from piece_movement import *
from castling_movement import *



# potential_castling = None


def verify_castling_in_two_steps(prev_state, initial_pos, final_pos, potential_castling=None):


    # Vérifier si c'est la deuxième partie d'un castling
    if potential_castling is not None:

        result = verify_second_part_castling(potential_castling, initial_pos, final_pos, prev_state)

        if result is not None:
            print("CASTLING IN TWO STEPS DETECTED !!!!!!!!!!!!!!!")

        return 'castling_confirmed', result  

    else:

        # Détecter un potentiel début de castling
        potential_castling_first_part = detect_first_part_castling(initial_pos, final_pos, prev_state)

        if potential_castling_first_part is not None:
            print("POTENTIAL CASTLING FIRST PART DETECTED")

            return 'potential_castling', potential_castling_first_part

    return None, None



def verify_movement(prev_state, initial_pos, final_pos):
    """
    Vérifie les mouvements possibles et détecte les potentiels débuts de castling
    """
    valid_movements = []

    color = prev_state[initial_pos]
    # piece_value = abs(prev_state[initial_pos])

    

    # Vérification des mouvements standards
    if final_pos in pawn_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_pawn")

    if final_pos in rook_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_rook")

    if final_pos in knight_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_knight")

    if final_pos in bishop_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_bishop")

    if final_pos in queen_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_queen")

    if final_pos in king_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_king")
        
    


    return valid_movements



potential_castling = None

def analyze_move(prev_state, curr_state, potential_castling=None):
    """
    Compare two successive chess board states and determine the move made.
    
    Args:
        prev_state: numpy array of previous board state
        curr_state: numpy array of current board state
    
    Returns:
        dict with:
            - valid (bool): if the move appears valid
            - move_type (str): 'move', 'capture', or 'invalid'
            - from_pos (tuple): starting position (row, col)
            - to_pos (tuple): ending position (row, col)
            - piece (int): piece value that was moved
            - captured (int): piece value that was captured (if any)
    """


    

    # Convert to numpy arrays if not already
    prev_state = np.array(prev_state)
    curr_state = np.array(curr_state)
    
    # Find differences between states
    diff = curr_state - prev_state
    changes = np.where(diff != 0)

    # print('changes', changes)
    
    # Get positions where changes occurred
    positions = list(zip(changes[0], changes[1]))

    print('Change positions', positions)

    
    # If no changes or more than 2 positions changed, invalid move
    if len(positions) < 2 :
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'No changes detected'
        }, potential_castling

    
    elif len(positions) > 2:

        if len(positions) == 4:

            print('Special case 4 positions : could be a castling')

            castling_movements = castling_movement(positions, prev_state, curr_state)

            print('castling_movements', castling_movements)

            if castling_movements['valid']:
                return castling_movements, potential_castling


        

        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'Invalid number of position changes'
        }, potential_castling
    

    ###########################################
    # * MOVEMENT ANALYSIS
    ###########################################

    if len(positions) == 2:

        print('MOVEMENT ANALYSIS')

        pos1, pos2 = positions
        
        initial_pos = None
        final_pos = None

        # La position initiale est vide après le mouvement
        if curr_state[pos1] == 0:
            initial_pos = pos1
            final_pos = pos2
        else:
            initial_pos = pos2
            final_pos = pos1

        moving_piece = prev_state[initial_pos]

        # print('moving_piece', moving_piece)
        
        # Position final avant le mouvement
        target_square = prev_state[final_pos]

        # print('targetl_value', target_value)

        # Simple move
        if target_square == 0:
            movement = 'move'
        else:
            movement = 'capture'

        ###########################################
        # * VALID MOVEMENT
        ###########################################

        if curr_state[final_pos] == prev_state[initial_pos] and curr_state[initial_pos] == 0:


            castling_result, potential_result = verify_castling_in_two_steps(prev_state, initial_pos, final_pos, potential_castling)

            if castling_result is not None:
                if castling_result == 'castling_confirmed':
                    return potential_result, None
                elif castling_result == 'potential_castling':
                    potential_castling = potential_result

            valid_pieces = verify_movement(prev_state, initial_pos, final_pos)

            print('valid_pieces', valid_pieces)

            if len(valid_pieces) > 0:

                return {
                    'valid': True,
                    'move_type': movement,
                    'from_pos': initial_pos,
                    'to_pos': final_pos,
                    'piece': moving_piece,
                    'valid_pieces': valid_pieces
                }, potential_castling
            else:
                return {
                    'valid': False,
                    'move_type': 'invalid',
                    'message': 'No valid piece found matching the movement'
                }, potential_castling
        
        else:
            return {
                'valid': False,
                'move_type': 'invalid',
                'message': 'Final piece is not the same as the initial piece'
            }, potential_castling
    
    return {
        'valid': False,
        'move_type': 'invalid',
        'message': 'Unrecognized move pattern'
    }, potential_castling