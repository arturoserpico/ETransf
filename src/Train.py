import Policy
from EGraph import *
from EGraphGen import *

def train(model: Policy.Policy,
          valuation_fun: Callable[[ExprNode], float],
          n_episodes: int, 
          data_set_settings: EGraphDatasetConfig, 
          gen_settings: ExprGenConfig, 
          n_threads: int = 1):
    pass