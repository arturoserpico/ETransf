import Policy
from EGraph import *
import Episode
import torch
from EGraphGen import *
import Episode
import Train
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

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
    rules=rules,
    operators=OT,
    d_model=96,
    n_layers=3,
    n_layers_class=2,
    n_heads=8,
    n_heads_class=4,
    n_hidden_per_arg=4,
    n_hidden_per_param=4
)

print("parameter info: ")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"trainable: {trainable:,}  frozen: {frozen:,}  total: {trainable+frozen:,}")

def valuation(expr: ExprNode) -> float:
    # example: count AST nodes as cost
    return 1 + sum(valuation(c) for c in expr.children)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

# --- fixed test expression: (x + 0) ---
expr = ExprNode(
    op_id    = OT.id("add"),
    children = [
        ExprNode(op_id=OpTable.OP_VAR,   param=ST.intern("x")),
        ExprNode(op_id=OpTable.OP_CONST, param=0),
    ]
)

print(f"test expression cost: {valuation(expr)}")
print(f"optimal cost: 1  (just x)\n")

# --- config ---
device = torch.device("cpu")
n_episodes  = 5000
batch_size  = 32
n_threads   = 24
log_every   = 1

def run_one(_) -> tuple[torch.Tensor, float]:
    runner = Episode.Runner(
        model          = model,
        valuation_fun  = valuation,
        step_penalty   = 0.05,
        growth_penalty = 0.0,
        exit_factor    = 5.0,
    )
    return runner.run_episode_train_ppo(
        expr       = expr,
        max_steps  = 200,
        gamma      = 0.99,
        clip_eps   = 0.2,
        entropy_coeff = 0.001,
        value_coeff   = 0.25,
        ppo_epochs    = 2,
        device        = device,
    )

# --- batched training loop ---
n_batches    = (n_episodes + batch_size - 1) // batch_size
reward_history: list[float] = []
loss_history:   list[float] = []

for batch in range(1, n_batches + 1):
    batch_losses:  list[torch.Tensor] = []
    batch_rewards: list[float]        = []

    for chunk_start in range(0, batch_size, n_threads):
        chunk_size = min(n_threads, batch_size - chunk_start)

        with ThreadPoolExecutor(max_workers=chunk_size) as pool:
            futures = [pool.submit(run_one, None) for _ in range(chunk_size)]
            for f in futures:
                loss, reward = f.result()
                batch_losses.append(loss)
                batch_rewards.append(reward)

    optimizer.zero_grad()
    avg_loss = torch.stack(batch_losses).mean()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    reward_history.extend(batch_rewards)
    loss_history.append(avg_loss.item())

    episode = batch * batch_size
    if batch % log_every == 0:
        avg_reward = sum(batch_rewards) / len(batch_rewards)
        print(
            f"episode {episode:>5}"
            f"  loss={avg_loss.item():+.4f}"
            f"  avg_reward={avg_reward:.4f}"
        )

# --- final check ---
print("\n--- final inference ---")
runner = Episode.Runner(
    model          = model,
    valuation_fun  = valuation,
    step_penalty   = 0.0,
    growth_penalty = 0.0,
    exit_factor    = 1.0,
)
result = runner.run_episode(expr, max_steps=50)
print(f"input:  (x + 0)  cost={valuation(expr)}")
print(f"output: {result}  cost={valuation(result)}")
print(f"solved: {valuation(result) == 1}")

torch.save(model.state_dict(), "data/model.pt")

plt.plot(loss_history)

plt.plot(reward_history)

plt.show()