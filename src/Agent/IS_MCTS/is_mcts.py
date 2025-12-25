from src.Interfaces import IAgent, IEnv
from src.common import LogData, Config

import GunjinShogiCore as GSC

from src.Agent.IS_MCTS.network import IsMctsNetwork

import numpy as np
import torch



class Node:
    def __init__(self, c=0.7, parent:"Node" = None, action = -1):
        max_nodes = Config.board_shape_int*Config.board_shape_int
        
        self.parent:"Node" = parent
        self.children: list["Node"] = [None]*max_nodes
        self.children_visits = np.zeros(max_nodes, np.int32)
        
        self.isnt_children = np.ones(max_nodes, np.bool)
        
        self.c = c

        self.expanded = False
        
        self.is_leaf = True
        
        self.wins = 0
        self.visited = 0
        
        self.children_ns = np.zeros(max_nodes, np.int32)
        self.children_xs = np.zeros(max_nodes, np.float32)
        
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
        max_nodes = Config.board_shape_int*Config.board_shape_int
        
        isnts = np.zeros(max_nodes, np.bool)
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
    def __init__(
        self, player: GSC.Player, c=0.7, sim_time = 100, 
        in_channels: int = 41, mid_channels: int = 20, model_path: str = "models/is_mcts/v2/model_100000.pth",
        device: torch.device = torch.device("cpu")
        ):
        self.player = player
        self.opponent = GSC.Player.PLAYER_TWO if player == GSC.Player.PLAYER_ONE else GSC.Player.PLAYER_ONE
        self.c = c
        self.sim_time = sim_time
        #self.network = IsMctsNetwork(in_channels, mid_channels).to(device)
        #self.network.load_state_dict(torch.load(model_path))
        #self.network.eval()
        self.device = device
        
    def simulation(self, env: IEnv) -> int:
        
        if(env.get_winner() == self.player): return 1
        elif(env.get_winner() == self.opponent): return 0
        elif(env.get_winner() == -1): return 0.5
        
        while(True):
            legals = env.legal_move()
            if(legals.size <= 0):
                env.step(-1)
                return 0
            
            action = np.random.choice(legals)
            _,_,frag = env.step(action)
            
            if(frag != GSC.BattleEndFrag.CONTINUE and frag != GSC.BattleEndFrag.DEPLOY_END):
                break
            
        if(env.get_winner() == self.player): return 1
        elif(env.get_winner() == self.opponent): return 0
        else: return 0.5
            
            
    
    def step(self, log:LogData, frag: GSC.BattleEndFrag):
        pass
    
    def reset(self):
        pass
        
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
        
        if(is_n1): 
            n1.update(self.simulation(env))
        else:
            pieces = np.arange(Config.piece_limit)
            np.random.shuffle(pieces)
            o_env = env.get_defined_env(pieces, player=self.opponent)
            n2.update(self.simulation(o_env))
    
    def get_action(self, env: IEnv) -> int:
        tree1 = Node(self.c)
        tree2 = Node(self.c)
        for i in range(self.sim_time):
            pieces1 = np.arange(Config.piece_limit)
            np.random.shuffle(pieces1)
            pieces2 = np.arange(Config.piece_limit)
            np.random.shuffle(pieces2)
        
            opponent = GSC.Player.PLAYER_ONE if self.player == GSC.Player.PLAYER_TWO else GSC.Player.PLAYER_TWO
        
            denv = env.get_defined_env(pieces1, player=self.player)
            denv2 = denv.get_defined_env(pieces2, player=opponent)
            
        
            self.search(tree1, tree2, denv)
            
            del denv, denv2
            
        return np.argmax(tree1.children_visits)
    
    def get_first_board(self) -> np.ndarray:
        pieces = np.arange(Config.piece_limit)
        np.random.shuffle(pieces)
        return pieces    
    
        