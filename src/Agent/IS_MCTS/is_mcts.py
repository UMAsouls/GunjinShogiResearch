from src.const import BOARD_SHAPE_INT, PIECE_LIMIT

from src.Interfaces import IAgent, IEnv
import GunjinShogiCore as GSC

import numpy as np

MAX_NODES = BOARD_SHAPE_INT*BOARD_SHAPE_INT

class Node:
    def __init__(self, c=0.7, parent:"Node" = None, action = -1):
        self.parent:"Node" = parent
        self.children: list["Node"] = [None]*MAX_NODES
        self.children_visits = np.zeros(MAX_NODES, np.int32)
        
        self.isnt_children = np.ones(MAX_NODES, np.bool)
        
        self.c = c

        self.expanded = False
        
        self.is_leaf = True
        
        self.wins = 0
        self.visited = 0
        
        self.children_ns = np.zeros(MAX_NODES, np.int32)
        self.children_xs = np.zeros(MAX_NODES, np.float32)
        
        self.ns_sum = 0
        
        self.action = action
        
    def visit(self) -> None:
        self.visited += 1
        self.parent.children_visits[self.action] += 1
        
    def get_child(self, action:int) -> "Node":
        node = self.children[action]
        node.visit()
        return node
    
    def select_by_opponent(self, action:int) -> "Node":
        if(self.children[action] is None): self.children[action] = Node(self.c,self,action)
        return self.get_child(action)
    
    def expand_legal(self, legals: np.typing.NDArray[np.int32]):
        isnts = np.zeros(MAX_NODES, np.bool)
        isnts[legals] = self.isnt_children[legals]
        none_indices = np.where(isnts)[0]
        for i in none_indices:
            self.children[i] = Node(self.c, self, i)
        
        self.isnt_children[none_indices] = np.bool(False)

    def select(self, legals: np.ndarray) -> "Node":
        self.expand_legal(legals)
        
        self.children_ns[legals] += 1
        self.ns_sum += legals.size
        
        ucbs = np.array([
            self.children_xs[legals] + self.c*np.sqrt(self.ns_sum/self.children_ns[legals]),
            legals
        ]).T
        
        sorted_indices = np.argsort(ucbs[:,0])
        sorted_ucbs = ucbs[sorted_indices]
        
        max_indices= np.where(sorted_ucbs[:,0]==sorted_ucbs[-1,0])[0]
        index = np.random.choice(max_indices)
        action = sorted_ucbs[index,1]
        return np.int32(action)

    def expand(self) -> None:
        self.is_leaf = False
        
    def update(self, win:int) -> None:
        self.wins += win
        if(self.parent == None): return
        self.parent.parent_update(self.wins/self.visited, self.action)
        self.parent.update(win)
        
    def parent_update(self, win_rate, action):
        self.children_xs[action] = win_rate
        
    
class ISMCTSAgent(IAgent):
    def __init__(self, player: GSC.Player, c=0.7, sim_time = 100):
        self.player = player
        self.opponent = GSC.Player.PLAYER_TWO if player == GSC.Player.PLAYER_ONE else GSC.Player.PLAYER_ONE
        self.c = c
        self.sim_time = sim_time
        
    def simulation(self, env: IEnv) -> int:
        while True:
            legals = env.legal_move()
            action = -1
            if(legals.size > 0) : action = np.random.choice(legals)
            _, _, frag = env.step(action)
            if(frag != GSC.BattleEndFrag.CONTINUE): break
            
        if(env.get_winner() == self.player): return 1
        else: return 0
        
    def search(self, root1: Node, root2: Node, env: IEnv):
        n1 = root1
        n2 = root2
        
        is_n1 = True
        while(not n1.is_leaf and not n2.is_leaf):
            node = n1 if is_n1 else n2
            o_node = n2 if is_n1 else n1
            
            legals = env.legal_move()
            if(legals.size <= 0):
                env.step(-1)
                break
            
            action = node.select(legals)
            node = node.get_child(action)
            o_node = o_node.select_by_opponent(action)
            env.step(action)
            
            n1 = node if is_n1 else o_node
            n2 = o_node if is_n1 else node
            
            is_n1 = not is_n1
        
        winner = env.get_winner()
        if(winner != None):
            if(is_n1): n1.update(winner == self.player)
            else: n2.update(winner == self.opponent)
            return
        
        n1.expand()
        n2.expand()
        
        if(is_n1): n1.update(self.simulation(env))
        else: n2.update(self.simulation(env))
    
    def get_action(self, env: IEnv) -> int:
        tree1 = Node(self.c)
        tree2 = Node(self.c)
        for i in range(self.sim_time):
            pieces1 = np.arange(PIECE_LIMIT)
            np.random.shuffle(pieces1)
            pieces2 = np.arange(PIECE_LIMIT)
            np.random.shuffle(pieces2)
        
            opponent = GSC.Player.PLAYER_ONE if self.player == GSC.Player.PLAYER_TWO else GSC.Player.PLAYER_TWO
        
            denv = env.get_defined_env(pieces1, player=self.player)
        
            self.search(tree1, tree2, denv)
            
            del denv
            
        return np.argmax(tree1.children_visits)
    
    def get_first_board(self) -> np.ndarray:
        pieces = np.arange(PIECE_LIMIT)
        np.random.shuffle(pieces)
        return pieces    
    
        