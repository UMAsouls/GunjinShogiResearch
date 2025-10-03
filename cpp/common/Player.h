#ifndef PLAYER_H
#define PLAYER_H

enum Player {
    PLAYER_ONE,
    PLAYER_TWO
};

Player getOpponent(Player player) {
    return player == Player::PLAYER_ONE ? Player::PLAYER_TWO : Player::PLAYER_ONE;
}

#endif