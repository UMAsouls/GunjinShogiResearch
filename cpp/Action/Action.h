#ifndef ACTION_H
#define ACTION_H

#include "common/Config.h"

struct Action
{
    int fromX;
    int fromY;
    int toX;
    int toY;
};

int MakeActionToInt(Action a, Config& c);

Action GetActionFromInt(int a, Config& c);


#endif