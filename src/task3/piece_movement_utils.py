


def is_on_board(position):
    return position[0] in range(8) and position[1] in range(8)


def next_cases(p1, p2, game_state, color, positions = []):

    # p3 = 2*p2 - p1

    p3 = (2*p2[0] - p1[0], 2*p2[1] - p1[1])

    if is_on_board(p3):

        if game_state[p3] == 0:
            positions.append(p3)
            next_cases(p2, p3, game_state, color,positions)

        elif game_state[p3] * color < 0:
            positions.append(p3)


    return positions

        



