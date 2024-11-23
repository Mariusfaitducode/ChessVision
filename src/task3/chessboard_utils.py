
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import copy



PIECES_TO_NUM = {
    "square": 0,
    "white_pawn": 1,
    "white_rook": 2,
    "white_knight": 3,
    "white_bishop": 4,
    "white_king": 5,
    "white_queen": 6,
    "white_unknown": 7,
    "black_pawn": -1,
    "black_rook": -2,
    "black_knight": -3,
    "black_bishop": -4,
    "black_king": -5,
    "black_queen": -6,
    "black_unknown": -7,
}

NUM_TO_PIECE = {
    0: "square",
    1: "white_pawn",
    2: "white_rook",
    3: "white_knight",
    4: "white_bishop",
    5: "white_king",
    6: "white_queen",
    7: "white_unknown",
    -1: "black_pawn",
    -2: "black_rook",
    -3: "black_knight",
    -4: "black_bishop",
    -5: "black_king",
    -6: "black_queen",
    -7: "black_unknown"
}


piece_images = {
    piece: Image.open(f'../../assets/{piece}.png') for piece in PIECES_TO_NUM.keys() if piece != "square"
}


def draw_chessboard(board, save=False, show=False):
    board = np.flipud(board)
    fig, ax = plt.subplots()

    chessboard_pattern = np.zeros((8, 8))
    chessboard_pattern[1::2, ::2] = 1
    chessboard_pattern[::2, 1::2] = 1

    ax.imshow(chessboard_pattern, cmap='gray', interpolation='none')

    for i in range(8):
        for j in range(8):
            piece_value = board[i, j]
            if piece_value != 0:
                piece_name = NUM_TO_PIECE[piece_value]
                piece_image = piece_images[piece_name]
                ax.imshow(piece_image, extent=[j - 0.5, j + 0.5, i - 0.5, i + 0.5])

    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    if save:
        fig.savefig(f"{save}.png")

    if show: plt.show()
    
    matplotlib.pyplot.close()



def modify_game_state(game_state, move):
    new_game_state = copy.copy(np.flipud(game_state)) # A1 is 0,0
    if " -> " in move:
        init_pos = move.split(" -> ")[0]
        new_pos = move.split(" -> ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif " + " in move:
        init_pos = move.split(" + ")[0]
        new_pos = move.split(" + ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif " x " in move:
        init_pos = move.split(" x ")[0]
        new_pos = move.split(" x ")[1]

        init_y_pos = int(init_pos[-1]) - 1
        init_x_pos = int(ord(init_pos[-2]) - 96 - 1)

        new_y_pos = int(new_pos[-1]) - 1
        new_x_pos = int(ord(new_pos[-2]) - 96 - 1)      

        piece = new_game_state[init_y_pos, init_x_pos]
        new_game_state[init_y_pos, init_x_pos] = 0
        new_game_state[new_y_pos, new_x_pos] = piece

    elif "O-O-O" in move:
        if move[0] == "w":
            new_game_state[0, 2] = new_game_state[0, 4]
            new_game_state[0, 3] = new_game_state[0, 0]
            new_game_state[0, 0] = 0
            new_game_state[0, 4] = 0
        else:
            new_game_state[7, 2] = new_game_state[7, 4]
            new_game_state[7, 3] = new_game_state[7, 0]
            new_game_state[7, 0] = 0
            new_game_state[7, 4] = 0

    elif "O-O" in move:
        if move[0] == "w":
            new_game_state[0, 6] = new_game_state[0, 4]
            new_game_state[0, 5] = new_game_state[0, 7]
            new_game_state[0, 7] = 0
            new_game_state[0, 4] = 0
        else:
            new_game_state[7, 6] = new_game_state[7, 4]
            new_game_state[7, 5] = new_game_state[7, 7]
            new_game_state[7, 7] = 0
            new_game_state[7, 4] = 0
        

    return np.flipud(new_game_state)