
from piece_movement_utils import *

import numpy as np


def pawn_movement(position, game_state):

    color = game_state[position]
    row, col = position
    possible_moves = []

    if (color > 0): # White pawn

        # Move forward
        if (game_state[row+1, col] == 0):
            possible_moves.append((row+1, col))

        # Move forward two squares on pawn first move
        if (game_state[row+2, col] == 0 and row == 1):
            possible_moves.append((row+2, col))
        
        # Capture diagonally
        if (is_on_board((row+1, col-1)) and game_state[row+1, col-1] < 0): 
            possible_moves.append((row+1, col-1))
        if (is_on_board((row+1, col+1)) and game_state[row+1, col+1] < 0):
            possible_moves.append((row+1, col+1))

    elif (color < 0): # Black pawn

        # Move forward
        if (game_state[row-1, col] == 0):
            possible_moves.append((row-1, col))

        # Move forward two squares on pawn first move
        if (game_state[row-2, col] == 0 and row == 6):
            possible_moves.append((row-2, col))
        
        # Capture diagonally
        if (is_on_board((row-1, col-1)) and game_state[row-1, col-1] > 0): 
            possible_moves.append((row-1, col-1))
        if (is_on_board((row-1, col+1)) and game_state[row-1, col+1] > 0):
            possible_moves.append((row-1, col+1))

    return possible_moves



def rook_movement(position, game_state):

    color = game_state[position]
    row, col = position
    possible_moves = []

    for i in range(-1, 2, 2):

        if is_on_board((row+i, col)):

            if game_state[row+i, col] == 0:
                possible_moves.append((row+i, col))

                next_possible_moves = next_cases(position, (row+i, col), game_state, color)

                possible_moves.extend(next_possible_moves)

            elif game_state[row+i, col] * color < 0:
                possible_moves.append((row+i, col))
            

        if is_on_board((row, col+i)):

            if game_state[row, col+i] == 0:
                possible_moves.append((row, col+i))

                next_possible_moves = next_cases(position, (row, col+i), game_state, color)
                
                possible_moves.extend(next_possible_moves)

            elif game_state[row, col+i] * color < 0:
                possible_moves.append((row, col+i))
    
    return possible_moves


def knight_movement(position, game_state):

    color = game_state[position]
    row, col = position
    possible_moves = []

    # Knight moves in L-shape: 2 squares in one direction and 1 square perpendicular
    knight_moves = [
        (-2, -1), (-2, 1),  # Up 2, left/right 1
        (2, -1), (2, 1),    # Down 2, left/right 1
        (-1, -2), (1, -2),  # Left 2, up/down 1
        (-1, 2), (1, 2)     # Right 2, up/down 1
    ]

    for move in knight_moves:
        new_row = row + move[0]
        new_col = col + move[1]
        
        if is_on_board((new_row, new_col)):
            # Can move if square is empty or contains enemy piece
            if game_state[new_row, new_col] == 0 or game_state[new_row, new_col] * color < 0:
                possible_moves.append((new_row, new_col))

    return possible_moves


def bishop_movement(position, game_state):
    
    color = game_state[position]
    row, col = position
    possible_moves = []

    for i in [-1, 1]:
        for j in [-1, 1]:

            if is_on_board((row+i, col+j)):

                if game_state[row+i, col+j] == 0:
                    possible_moves.append((row+i, col+j))

                    next_possible_moves = next_cases(position, (row+i, col+j), game_state, color)

                    possible_moves.extend(next_possible_moves)


                elif game_state[row+i, col+j] * color < 0:
                    possible_moves.append((row+i, col+j))
    
    
    return possible_moves



def queen_movement(position, game_state):
    
    return rook_movement(position, game_state) + bishop_movement(position, game_state)



def king_movement(position, game_state):
    
    color = game_state[position]
    row, col = position
    possible_moves = []

    for i in range(-1, 2):
        for j in range(-1, 2):

            if is_on_board((row+i, col+j)):
                if game_state[row+i, col+j] == 0 or game_state[row+i, col+j] * color < 0:
                    possible_moves.append((row+i, col+j))
    
    return possible_moves



def castling_movement(positions, prev_state, curr_state):

    # ! Array of game in bad format -> A1 = H1 -> H = 0 and E = 3
    # Apply vertical symmetry to correct board orientation (A1 should be (0,0) not (0,7))
    # prev_state = np.fliplr(prev_state.copy())
    # curr_state = np.fliplr(curr_state.copy())
    
    # Apply same transformation to positions array
    # positions = [(pos[0], 7-pos[1]) for pos in positions]

    print('Castling positions', positions)


    # King positons must be E1 or E8
    king_positions = [(0, 4), (7, 4)]

    row = None

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
        return {
            'valid': False,
            'move_type': 'invalid',
            'message': 'King position not found'
        }
    
    rook_positions = [(row, 0), (row, 7)]
    # rook_position = None

    # Verify if the left rook changed was in the good position and with the right color
    if rook_positions[0] in positions and prev_state[rook_positions[0]]*color > 0:
        
        # Left castling
        king_final = (row, 2)
        rook_final = (row, 3)

        # Verify if the castling move was valid
        if prev_state[rook_final] == 0 and prev_state[king_final] == 0:

            return {
                'valid': True,
                'move_type': 'castling',
                'message': 'Queenside castling move is valid',
                'king_final': king_final,
                'rook_final': rook_final,
                'color': color
            }

    # Verify if the right rook changed was in the good position and with the right color
    elif rook_positions[1] in positions and prev_state[rook_positions[1]]*color > 0:

        # Right castling
        king_final = (row, 6)
        rook_final = (row, 5)

        # Verify if the castling move was valid
        if prev_state[rook_final] == 0 and prev_state[king_final] == 0:

            return {
                'valid': True,
                'move_type': 'castling',
                'message': 'Kingside castling move is valid',
                'king_final': king_final,
                'rook_final': rook_final,
                'color': color
            }

    return {
        'valid': False,
        'move_type': 'invalid',
        'message': 'Castling move is invalid'
    }



