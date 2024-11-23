
from piece_movement_utils import *


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

    pass

