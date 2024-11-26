
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






