from src.const import BOARD_SHAPE, BOARD_SHAPE_INT

from src.Agent import RandomAgent
from src.VS import Agent_VS

from src.GunjinShogi import Environment, JudgeBoard, TensorBoard

import torch

BATTLES = 100

LOG_NAME = "random_test_1"

def main():
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    
    judgeboard = JudgeBoard(BOARD_SHAPE)
    tensorboard = TensorBoard(BOARD_SHAPE, torch.device("cpu"))
    
    env = Environment(judgeboard, tensorboard)
    
    wins1 = 0
    wins2 = 0
    
    for i in range(BATTLES):
        win = Agent_VS(agent1, agent2, env, LOG_NAME)
        if(win == 1): wins1 += 1
        elif(win == 2): wins2 += 1
        
        print(f"BattleEnds: {i}/{BATTLES}")
        
    print(f"agent1: {wins1}回, agent2: {wins2}回")
    
if __name__ == "__main__":
    main()