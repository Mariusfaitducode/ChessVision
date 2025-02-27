import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from task3.piece_movement import *
from task3.castling_movement import *



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

    if final_pos in pawn_movement(initial_pos, prev_state):
        valid_movements.append(f"{'white' if color > 0 else 'black'}_pawn")


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

            potential_pieces = verify_movement(prev_state, initial_pos, final_pos)

            previous_pieces = game_actualization['piece_certainty'][initial_pos]

            print('potential_pieces', potential_pieces)
            print('previous_pieces', previous_pieces)

            valid_pieces = []
            for piece in potential_pieces:
                if piece in previous_pieces:
                    valid_pieces.append(piece)

            print('valid_pieces', valid_pieces)

            if len(potential_pieces) == 1:
                valid_movements.append((combination, potential_pieces))
                print(f"Automatically considering piece: {potential_pieces[0]} for {combination}")
            elif len(valid_pieces) > 0:
                valid_movements.append((combination, valid_pieces))

        print('valid_movements', valid_movements)

        if len(valid_movements) == 1:

            print('1 VALID MOVEMENT DETECTED')

            combination = valid_movements[0][0]
            valid_pieces = valid_movements[0][1]

            move = 'move' if prev_state[combination[1]] == 0 else 'capture'

            # Trouver la position qui n'est pas dans la combinaison valide
            error_pos = None
            for pos in positions:
                if pos not in combination:
                    error_pos = pos
                    break

            if len(potential_pieces) == 1:
                return {
                    'valid': True,
                    'move_type': move,
                    'from_pos': combination[0],
                    'to_pos': combination[1],
                    'piece': prev_state[combination[0]],
                    'valid_pieces': potential_pieces,
                    'error_pos': error_pos
                }, potential_castling
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

        # elif len(valid_movements) > 1:
        #         # Prioritize moves based on the piece's value or type, accounting for both colors
        #     def prioritize_moves(movements, prev_state):
        #         # Define priority values for both white and black pieces
        #         piece_priority = {
        #             'white_king': 1,
        #             'white_queen': 2,
        #             'white_rook': 3,
        #             'white_bishop': 4,
        #             'white_knight': 5,
        #             'white_pawn': 6,
        #             'black_king': 1,
        #             'black_queen': 2,
        #             'black_rook': 3,
        #             'black_bishop': 4,
        #             'black_knight': 5,
        #             'black_pawn': 6
        #         }

        #         def get_piece_name(position):
        #             piece_value = prev_state[position]
        #             if piece_value > 0:
        #                 color = 'white'
        #             else:
        #                 color = 'black'
        #             piece_type = {
        #                 1: 'pawn',
        #                 2: 'knight',
        #                 3: 'bishop',
        #                 4: 'rook',
        #                 5: 'queen',
        #                 6: 'king'
        #             }.get(abs(piece_value), 'unknown')
        #             return f"{color}_{piece_type}"

        #         prioritized = sorted(
        #             movements,
        #             key=lambda mv: min(
        #                 piece_priority.get(get_piece_name(mv[0][0]), 10)
        #                 for piece in mv[1]
        #             )
        #         )
        #         return prioritized[0]  # Select the highest-priority move

        #     # Select the best move based on priority
        #     best_move = prioritize_moves(valid_movements, prev_state)
        #     combination = best_move[0]
        #     valid_pieces = best_move[1]

        #     move = 'move' if prev_state[combination[1]] == 0 else 'capture'

        #     error_pos = None
        #     for pos in positions:
        #         if pos not in combination:
        #             error_pos = pos
        #             break

        #     return {
        #         'valid': True,
        #         'move_type': move,
        #         'from_pos': combination[0],
        #         'to_pos': combination[1],
        #         'piece': prev_state[combination[0]],
        #         'valid_pieces': valid_pieces,
        #         'error_pos': error_pos
        #     }, potential_castling

        else:
            return {
                'valid': False,
                'move_type': 'invalid',
                'message': 'Too much positions changed'
            }, potential_castling

    else:
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'Too much positions changed'
        }, potential_castling
    
    # return {
    #     'valid': False,
    #     'move_type': 'invalid',
    #     'message': 'Unrecognized move pattern'
    # }, potential_castling