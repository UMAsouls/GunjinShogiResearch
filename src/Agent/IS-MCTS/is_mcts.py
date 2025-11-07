from src.const import BOARD_SHAPE_INT

import numpy as np

MAX_NODES = BOARD_SHAPE_INT*BOARD_SHAPE_INT

class Node:
    def __init__(self, parent:"Node" = None):
        self.parent:"Node" = parent
        self.children = [None]*MAX_NODES
        self.children_id = np.array([])

        self.children_ucb = np.zeros(MAX_NODES, np.float32)

        self.expanded = False

    def select(self) -> "Node":
        pass

    def select_by_opponent(self, action: int) -> "Node":
        pass

    def expand(self, actions: int) -> None:
        pass