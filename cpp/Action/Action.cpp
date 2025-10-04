#include "Action/Action.h"

Action GetActionFromInt(int action, Config& c) {
    int width = c.getBoardShape().first;
    int height = c.getBoardShape().second;
    int shape = width*height;

    int from = action / shape;
    int to = action % shape;

    return Action{
        from%width,
        from/width,
        to%width,
        to/width
    };
}

int MakeActionToInt(Action action, Config& c) {
    int fx = action.fromX;
    int fy = action.fromY;
    int tx = action.toX;
    int ty = action.toY;

    int width = c.getBoardShape().first;
    int height = c.getBoardShape().second;
    int shape = width*height;

    return (fy*width + fx)*shape + (ty*width + tx);
}