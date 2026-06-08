import threading
from concurrent.futures import ThreadPoolExecutor
from Episode import *
import os

def train(
    model:          Policy.Policy,
    expr_cfg:       ExprGenConfig,
    valuation_fun:  Callable[[ExprNode], float],
    rules:          list[RewriteRule],
    n_episodes:     int   = 10_000,
    lr:             float = 3e-4,
    gamma:          float = 0.99,
    entropy_coeff:  float = 0.01,
    max_steps:      int   = 500,
    log_every:      int   = 50,
    step_penalty:   float = 0.01,
    growth_penalty: float = 0.01,
    exit_factor:    float = 1.0,
    seed:           int | None = None,
    start_depth:    int   = 1,
    max_depth:      int   = 6,
    depth_interval: int   = 1000,
    batch_size:     int   = 16,   # episodes per gradient update
    n_threads:      int   = 4,    # parallel workers
    checkpoint_every: int        = 500,
    checkpoint_path:  str        = "checkpoint.pt",
    resume_from:      str | None = None,
):
    device = next(model.parameters()).device
    print(f"training on {device} — batch_size={batch_size}  n_threads={n_threads}")

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    reward_history: list[float] = []
    loss_history:   list[float] = []

    start_batch = 1
    if resume_from is not None:
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        reward_history     = checkpoint["reward_history"]
        loss_history       = checkpoint["loss_history"]
        start_batch        = checkpoint["batch"] + 1
        expr_cfg.max_depth = checkpoint["depth"]
        print(f"resumed from batch {start_batch}, depth={expr_cfg.max_depth}")

    expr_cfg.max_depth = start_depth

    def run_one(expr: ExprNode) -> tuple[torch.Tensor, float]:
        runner = Runner(
            model          = model,
            valuation_fun  = valuation_fun,
            step_penalty   = step_penalty,
            growth_penalty = growth_penalty,
            exit_factor    = exit_factor,
        )
        return runner.run_episode_train(
            expr          = expr,
            max_steps     = max_steps,
            gamma         = gamma,
            entropy_coeff = entropy_coeff,
            device        = device,
        )

    n_batches = (n_episodes + batch_size - 1) // batch_size

    for batch in range(start_batch, n_batches + 1):
        episode_start = (batch - 1) * batch_size + 1
        episode_end   = min(batch * batch_size, n_episodes)
        n_this_batch  = episode_end - episode_start + 1

        # --- curriculum ---
        target_depth = min(start_depth + (episode_start - 1) // depth_interval, max_depth)
        if expr_cfg.max_depth != target_depth:
            expr_cfg.max_depth = target_depth
            print(f"\n[curriculum] batch {batch}: max_depth → {target_depth}")

        exprs = [random_expr(expr_cfg) for _ in range(n_this_batch)]

        # --- collect episodes in chunks of n_threads ---
        batch_losses:  list[torch.Tensor] = []
        batch_rewards: list[float]        = []

        for chunk_start in range(0, n_this_batch, n_threads):
            chunk = exprs[chunk_start : chunk_start + n_threads]

            with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
                futures = [pool.submit(run_one, expr) for expr in chunk]
                for f in futures:
                    loss, reward = f.result()
                    batch_losses.append(loss)
                    batch_rewards.append(reward)

        # --- single update over full batch ---
        optimizer.zero_grad()
        avg_loss = torch.stack(batch_losses).mean()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        reward_history.extend(batch_rewards)
        loss_history.append(avg_loss.item())

        if batch % log_every == 0:
            avg_reward = sum(batch_rewards) / len(batch_rewards)
            print(
                f"batch {batch:>6}/{n_batches}"
                f"  episodes={episode_end}"
                f"  depth={expr_cfg.max_depth}"
                f"  loss={avg_loss.item():.4f}"
                f"  avg_reward={avg_reward:.4f}"
                f"  lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if batch % checkpoint_every == 0:
            dir_name = os.path.dirname(checkpoint_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            torch.save({
                "batch":          batch,
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "reward_history": reward_history,
                "loss_history":   loss_history,
                "depth":          expr_cfg.max_depth,
            }, checkpoint_path)
            print(f"  [checkpoint saved at batch {batch}]")

    return reward_history, loss_history