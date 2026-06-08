import Policy
from EGraph import *
from EGraphGen import *

EXIT = 0
SELECT = 1
NONE = 2

class Runner:
    def __init__(self, 
                 model: Policy.Policy, 
                 valuation_fun: Callable[[ExprNode], float],
                 step_penalty: float = None,
                 growth_penalty: float = None,
                 exit_factor: float = None):
        self.model = model
        self.valuation = valuation_fun
        self.step_penalty = step_penalty
        self.growth_penalty = growth_penalty
        self.exit_factor = exit_factor

    def init_episode(self, expr: ExprNode):
        self.step_count = 0
        self.initial = self.valuation(expr)
        self.current_best = self.valuation(expr)
        self.eg = EGraph()
        self.eclass_id = insert_expr(expr, self.eg)

    def step(self, actions: list[int]) -> float:
        new_best = self.eg.extract(self.eclass_id, self.valuation)

        if EXIT in actions:
            return (self.initial - new_best) * self.exit_factor

        if SELECT in actions:
            expr = self.eg.extract(self.eclass_id, self.valuation)
            
            
            self.eg = EGraph()


            reward = (self.initial - new_best) - self.step_penalty * self.step_count
            self.initial = new_best
            self.

        

    #def run_episode(model: Policy.Policy, eg: EGraph, valuation_fun: Callable[[ExprNode], float]):
    