import Policy
from EGraph import *
import Episode
import torch
from EGraphGen import *
import Episode
import Train
import matplotlib.pyplot as plt

OT = make_math_optable()
ST = SymTable()

var_ids = [ST.intern(v) for v in ["x", "y", "z"]]

expr_cfg = ExprGenConfig(
    ops         = [(OT.id("add"), 2), (OT.id("mul"), 2), (OT.id("sub"), 2), (OT.id("div"), 2)],
    var_ids     = var_ids,
    const_range = (0, 9),
    min_depth   = 1,
    max_depth   = 4,
    leaf_var_prob = 0.6,
)

rules = [
    # --- commutativity ---
    RewriteRule("add-comm",
        lhs=opat(OT, "add", mvar("x"), mvar("y")),
        rhs=opat(OT, "add", mvar("y"), mvar("x"))),
    RewriteRule("mul-comm",
        lhs=opat(OT, "mul", mvar("x"), mvar("y")),
        rhs=opat(OT, "mul", mvar("y"), mvar("x"))),

    # --- associativity ---
    RewriteRule("add-assoc-l",
        lhs=opat(OT, "add", opat(OT, "add", mvar("x"), mvar("y")), mvar("z")),
        rhs=opat(OT, "add", mvar("x"), opat(OT, "add", mvar("y"), mvar("z")))),
    RewriteRule("add-assoc-r",
        lhs=opat(OT, "add", mvar("x"), opat(OT, "add", mvar("y"), mvar("z"))),
        rhs=opat(OT, "add", opat(OT, "add", mvar("x"), mvar("y")), mvar("z"))),
    RewriteRule("mul-assoc-l",
        lhs=opat(OT, "mul", opat(OT, "mul", mvar("x"), mvar("y")), mvar("z")),
        rhs=opat(OT, "mul", mvar("x"), opat(OT, "mul", mvar("y"), mvar("z")))),
    RewriteRule("mul-assoc-r",
        lhs=opat(OT, "mul", mvar("x"), opat(OT, "mul", mvar("y"), mvar("z"))),
        rhs=opat(OT, "mul", opat(OT, "mul", mvar("x"), mvar("y")), mvar("z"))),

    # --- identity ---
    RewriteRule("add-zero-r",
        lhs=opat(OT, "add", mvar("x"), cpat(0)),
        rhs=mvar("x")),
    RewriteRule("add-zero-l",
        lhs=opat(OT, "add", cpat(0), mvar("x")),
        rhs=mvar("x")),
    RewriteRule("mul-one-r",
        lhs=opat(OT, "mul", mvar("x"), cpat(1)),
        rhs=mvar("x")),
    RewriteRule("mul-one-l",
        lhs=opat(OT, "mul", cpat(1), mvar("x")),
        rhs=mvar("x")),
    RewriteRule("sub-zero",
        lhs=opat(OT, "sub", mvar("x"), cpat(0)),
        rhs=mvar("x")),

    # --- annihilation ---
    RewriteRule("mul-zero-r",
        lhs=opat(OT, "mul", mvar("x"), cpat(0)),
        rhs=cpat(0)),
    RewriteRule("mul-zero-l",
        lhs=opat(OT, "mul", cpat(0), mvar("x")),
        rhs=cpat(0)),

    # --- negation ---
    RewriteRule("neg-neg",
        lhs=opat(OT, "neg", opat(OT, "neg", mvar("x"))),
        rhs=mvar("x")),
    RewriteRule("neg-zero",
        lhs=opat(OT, "neg", cpat(0)),
        rhs=cpat(0)),
    RewriteRule("sub-as-add-neg",
        lhs=opat(OT, "sub", mvar("x"), mvar("y")),
        rhs=opat(OT, "add", mvar("x"), opat(OT, "neg", mvar("y")))),
    RewriteRule("add-neg-as-sub",
        lhs=opat(OT, "add", mvar("x"), opat(OT, "neg", mvar("y"))),
        rhs=opat(OT, "sub", mvar("x"), mvar("y"))),
    RewriteRule("neg-mul",
        lhs=opat(OT, "neg", opat(OT, "mul", mvar("x"), mvar("y"))),
        rhs=opat(OT, "mul", opat(OT, "neg", mvar("x")), mvar("y"))),
    RewriteRule("mul-neg",
        lhs=opat(OT, "mul", opat(OT, "neg", mvar("x")), mvar("y")),
        rhs=opat(OT, "neg", opat(OT, "mul", mvar("x"), mvar("y")))),
    RewriteRule("neg-add",
        lhs=opat(OT, "neg", opat(OT, "add", mvar("x"), mvar("y"))),
        rhs=opat(OT, "add", opat(OT, "neg", mvar("x")), opat(OT, "neg", mvar("y")))),

    # --- subtraction ---
    RewriteRule("sub-self",
        lhs=opat(OT, "sub", mvar("x"), mvar("x")),
        rhs=cpat(0)),
    RewriteRule("sub-neg",
        lhs=opat(OT, "sub", cpat(0), mvar("x")),
        rhs=opat(OT, "neg", mvar("x"))),

    # --- distributivity ---
    RewriteRule("mul-add-distr-l",
        lhs=opat(OT, "mul", mvar("x"), opat(OT, "add", mvar("y"), mvar("z"))),
        rhs=opat(OT, "add", opat(OT, "mul", mvar("x"), mvar("y")), opat(OT, "mul", mvar("x"), mvar("z")))),
    RewriteRule("mul-add-distr-r",
        lhs=opat(OT, "add", opat(OT, "mul", mvar("x"), mvar("y")), opat(OT, "mul", mvar("x"), mvar("z"))),
        rhs=opat(OT, "mul", mvar("x"), opat(OT, "add", mvar("y"), mvar("z")))),
    RewriteRule("mul-sub-distr-l",
        lhs=opat(OT, "mul", mvar("x"), opat(OT, "sub", mvar("y"), mvar("z"))),
        rhs=opat(OT, "sub", opat(OT, "mul", mvar("x"), mvar("y")), opat(OT, "mul", mvar("x"), mvar("z")))),
    RewriteRule("mul-sub-distr-r",
        lhs=opat(OT, "sub", opat(OT, "mul", mvar("x"), mvar("y")), opat(OT, "mul", mvar("x"), mvar("z"))),
        rhs=opat(OT, "mul", mvar("x"), opat(OT, "sub", mvar("y"), mvar("z")))),

    # --- division ---
    RewriteRule("div-one",
        lhs=opat(OT, "div", mvar("x"), cpat(1)),
        rhs=mvar("x")),
    RewriteRule("div-self",
        lhs=opat(OT, "div", mvar("x"), mvar("x")),
        rhs=cpat(1)),
    RewriteRule("div-zero-num",
        lhs=opat(OT, "div", cpat(0), mvar("x")),
        rhs=cpat(0)),
    RewriteRule("cancel-mul-div-r",
        lhs=opat(OT, "div", opat(OT, "mul", mvar("x"), mvar("y")), mvar("y")),
        rhs=mvar("x")),
    RewriteRule("cancel-mul-div-l",
        lhs=opat(OT, "div", opat(OT, "mul", mvar("x"), mvar("y")), mvar("x")),
        rhs=mvar("y")),
    RewriteRule("div-mul-cancel",
        lhs=opat(OT, "mul", opat(OT, "div", mvar("x"), mvar("y")), mvar("y")),
        rhs=mvar("x")),
]

model = Policy.Policy(
    rules          = rules,
    d_model        = 96,
    n_layers       = 4,
    n_layers_node  = 2,
    n_layers_class = 2,
    n_heads        = 4,
    n_heads_node   = 4,
    n_heads_class  = 4,
    max_arity      = 2,
    op_count       = len(OT)
)

print("parameter info: ")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:,}  frozen: {frozen:,}  total: {trainable+frozen:,}")

def valuation(expr: ExprNode) -> float:
    # example: count AST nodes as cost
    return 1 + sum(valuation(c) for c in expr.children)

reward_history, loss_history = Train.train(
    model         = model,
    expr_cfg      = expr_cfg,
    valuation_fun = valuation,
    rules         = rules,
    n_episodes    = 30_000,
    max_steps     = 200,
    log_every     = 1,
    depth_interval= 5000,
    start_depth=3,
    max_depth=10,
    lr=1e-7,
    step_penalty=0,
    exit_factor=0.5,
    batch_size=128,
    n_threads=24
)

torch.save(model.state_dict(), "data/model.pt")

plt.plot(loss_history)

plt.show()