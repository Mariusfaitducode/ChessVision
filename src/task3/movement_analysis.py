
import numpy as np

from piece_movement import *


def verify_movement(prev_state, initial_pos, final_pos):

    print("VERIFY MOVEMENT")

    valid_movements = []

    pawn_movements = pawn_movement(initial_pos, prev_state)

    print('pawn_movements', pawn_movements)
    print('final_pos', final_pos)

    if final_pos in pawn_movements:
        print('pawn move verified')
        valid_movements.append('pawn')


    rook_movements = rook_movement(initial_pos, prev_state)

    print('rook_movements', rook_movements)

    if final_pos in rook_movements:
        print('rook move verified')
        valid_movements.append('rook')
    
    return valid_movements



def analyze_move(prev_state, curr_state):
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


    print('MOVEMENT ANALYSIS')

    # Convert to numpy arrays if not already
    prev_state = np.array(prev_state)
    curr_state = np.array(curr_state)
    
    # Find differences between states
    diff = curr_state - prev_state
    changes = np.where(diff != 0)

    print('changes', changes)
    
    # Get positions where changes occurred
    positions = list(zip(changes[0], changes[1]))

    print('positions', positions)

    
    # If no changes or more than 2 positions changed, invalid move
    if len(positions) == 0 :
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'No changes detected'
        }
    
    elif len(positions) > 2:
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'Invalid number of position changes'
        }
    
    # For a basic move (no capture)
    if len(positions) == 2:


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

        print('moving_piece', moving_piece)
        
        # Position final avant le mouvement
        initial_final_value = prev_state[final_pos]

        print('initial_final_value', initial_final_value)

        from_square = f"{chr(97 + initial_pos[1])}{8 - initial_pos[0]}"
        to_square = f"{chr(97 + final_pos[1])}{8 - final_pos[0]}"

        # Simple move
        if initial_final_value == 0:

            move = f"{from_square} -> {to_square}"

            print('move', move)

            # TODO: Check if the move is valid

            valid = verify_movement(prev_state, initial_pos, final_pos)

            # new_gs = modify_game_state(prev_state, move)

            # diff = curr_state - new_gs
            # changes = np.where(diff != 0)

            # print('valid game state', len(changes))

            return {
                'valid': True,
                'move_type': 'move',
                'from_pos': initial_pos,
                'to_pos': final_pos,
                'piece': moving_piece,
            }

        # Capture
        else:

            move = f"{from_square} x {to_square}"

            print('move', move)

            # new_gs = modify_game_state(prev_state, move)

            # diff = curr_state - new_gs
            # changes = np.where(diff != 0)

            # print('valid game state', len(changes))

            return {
                'valid': True,
                'move_type': 'capture',
                'from_pos': initial_pos,
                'to_pos': final_pos,
                'piece': moving_piece
            }

            pass
    
    return {
        'valid': False,
        'move_type': 'invalid',
        'message': 'Unrecognized move pattern'
    }