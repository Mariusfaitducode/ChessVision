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

def analyze_move(prev_state, curr_state, potential_castling=None, game_actualization=None):
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
    if len(positions) == 0 :
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'No changes detected'
        }, potential_castling
    
    if len(positions) == 1 :
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'Only 1 change so its an error'
        }, potential_castling
    
    if len(positions) == 4:

        print('Special case 4 positions : could be a castling')

        castling_movements = castling_movement(positions, prev_state, curr_state)

        print('castling_movements', castling_movements)

        if castling_movements['valid']:
            return castling_movements, potential_castling

        # return {
        #     'valid': False,
        #     'move_type': 'invalid',
        #     'message': 'Invalid number of position changes'
        # }, potential_castling
    

    ###########################################
    # * MOVEMENT ANALYSIS
    ###########################################

    if len(positions) == 2 or len(positions) == 3:

        print('MOVEMENT ANALYSIS')
        
        initial_pos = []
        final_pos = []

        # * On sépare les positions initiales et finales
        for pos in positions:
            if curr_state[pos] == 0:
                initial_pos.append(pos)
            else:
                final_pos.append(pos)


        # * On retrouve les combinaisons valides
        valid_combinations = []

        for initial in initial_pos:
            for final in final_pos:
                if curr_state[final] == prev_state[initial]:
                    valid_combinations.append((initial, final))


        valid_movements = []
        for combination in valid_combinations:
            print('combination', combination)

            initial_pos = combination[0]
            final_pos = combination[1]

            castling_result, potential_result = verify_castling_in_two_steps(prev_state, initial_pos, final_pos, potential_castling)

            if castling_result is not None:
                if castling_result == 'castling_confirmed':
                    return potential_result, None
                elif castling_result == 'potential_castling':
                    potential_castling = potential_result

            valid_pieces = verify_movement(prev_state, initial_pos, final_pos)

            if len(valid_pieces) > 0:
                valid_movements.append((combination, valid_pieces))

        print('valid_movements', valid_movements)

        if len(valid_movements) == 1:

            combination = valid_movements[0][0]
            valid_pieces = valid_movements[0][1]

            move = 'move' if prev_state[combination[1]] == 0 else 'capture'

            # Trouver la position qui n'est pas dans la combinaison valide
            error_pos = None
            for pos in positions:
                if pos not in combination:
                    error_pos = pos
                    break

            # # Correct the error
            # curr_state[error_pos] = prev_state[error_pos]

            return {
                'valid': True,
                'move_type': move,
                'from_pos': combination[0],
                'to_pos': combination[1],
                'piece': prev_state[combination[0]],
                'valid_pieces': valid_pieces,
                'error_pos': error_pos
            }, potential_castling
        
        elif len(valid_movements) > 1:
            return {
                'valid': False,
                'move_type': 'invalid',
                'message': 'Too much valid movements found'
            }, potential_castling
        

    else:
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'Too much positions changed'
        }, potential_castling
    
    return {
        'valid': False,
        'move_type': 'invalid',
        'message': 'Unrecognized move pattern'
    }, potential_castling