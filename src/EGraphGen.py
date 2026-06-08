from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

# Assumes egraph.py (the implementation you provided) is importable.
from EGraph import (
    EGraph, ENode, EClass, ExprNode,
    OpTable, SymTable,
    make_var, make_const, make_op,
    RewriteRule,
)


def insert_expr(node: ExprNode, eg: EGraph) -> int:
    """Recursively insert an ExprNode tree into *eg*; return root e-class ID."""
    # Leaves: pass op_id and param straight through — identical to ENode construction.
    if not node.children:
        return eg.add(ENode(op=node.op_id, children=(), param=node.param))
    child_ids = [insert_expr(c, eg) for c in node.children]
    return eg.add(ENode(op=node.op_id, children=tuple(child_ids), param=0))


# ===========================================================================
# Random Expression Generator
# ===========================================================================

@dataclass
class ExprGenConfig:
    """
    All knobs for random expression generation.

    Attributes
    ----------
    ops          : list of (op_id, arity) for interior nodes — use OT.id("name")
                   to fill this after building your OpTable.
    var_ids      : pool of symbol IDs (from SymTable.intern) to draw from.
    const_range  : (lo, hi) inclusive range for random integer constants.
    min_depth    : minimum AST depth (0 = a single leaf).
    max_depth    : maximum AST depth.
    leaf_var_prob: probability that a leaf is a variable (vs a constant).
    """
    ops:           list[tuple[int, int]]   # (op_id, arity)
    var_ids:       list[int]               # symbol IDs
    const_range:   tuple[int, int] = (0, 9)
    min_depth:     int             = 1
    max_depth:     int             = 4
    leaf_var_prob: float           = 0.6


def random_expr(cfg: ExprGenConfig, depth: int = 0) -> ExprNode:
    """Recursively build a random ExprNode using integer op IDs throughout."""
    force_leaf = (depth >= cfg.max_depth)
    force_int  = (depth < cfg.min_depth)

    if force_leaf or (not force_int and random.random() < 0.35):
        # Leaf: var or const
        if random.random() < cfg.leaf_var_prob and cfg.var_ids:
            sym_id = random.choice(cfg.var_ids)
            return ExprNode(op_id=OpTable.OP_VAR, param=sym_id)
        else:
            val = random.randint(*cfg.const_range)
            return ExprNode(op_id=OpTable.OP_CONST, param=val)

    # Interior node
    op_id, arity = random.choice(cfg.ops)
    children = [random_expr(cfg, depth + 1) for _ in range(arity)]
    return ExprNode(op_id=op_id, children=children)


# ===========================================================================
# EGraph Dataset Generator
# ===========================================================================

@dataclass
class EGraphDatasetConfig:
    """
    Top-level configuration for dataset generation.

    Attributes
    ----------
    optable      : pre-built OpTable (must contain all op IDs in expr_cfg.ops).
    symtable     : shared SymTable (variables are interned here).
    expr_cfg     : expression generation parameters.
    rules        : rewrite rules to apply before returning each EGraph
                   (leave empty [] to return the initial graph with no rewrites).
    apply_rules  : if True, run apply_rules(rules) on each EGraph before
                   adding it to the dataset.
    deduplicate  : if True, skip EGraphs whose initial expression is
                   structurally identical to one already generated.
    seed         : random seed (None = non-deterministic).
    """
    optable:     OpTable
    symtable:    SymTable
    expr_cfg:    ExprGenConfig
    rules:       list[RewriteRule] = field(default_factory=list)
    apply_rules: bool              = False
    deduplicate: bool              = True
    seed:        Optional[int]     = None


def generate_egraphs(
    cfg: EGraphDatasetConfig,
    n:   int,
    *,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[EGraph]:
    """
    Generate *n* diverse EGraph objects according to *cfg*.

    Parameters
    ----------
    cfg          : dataset configuration.
    n            : number of EGraph objects to produce.
    on_progress  : optional callback(current, total) called every 100 graphs.

    Returns
    -------
    list[EGraph] of length *n*.
    """
    if cfg.seed is not None:
        random.seed(cfg.seed)

    seen_exprs: set[str]   = set()
    dataset:    list[EGraph] = []

    attempts     = 0
    max_attempts = n * 20

    while len(dataset) < n and attempts < max_attempts:
        attempts += 1

        # 1. Generate a random expression tree (all integer IDs, no strings)
        expr = random_expr(cfg.expr_cfg)

        # 2. (Optional) deduplicate by canonical integer s-expression
        if cfg.deduplicate:
            key = _sexpr(expr)
            if key in seen_exprs:
                continue
            seen_exprs.add(key)

        # 3. Build EGraph — no OpTable needed, op IDs go straight through
        eg = EGraph()
        insert_expr(expr, eg)

        # 4. (Optional) apply rewrite rules
        if cfg.apply_rules and cfg.rules:
            eg.apply_rules(cfg.rules)

        dataset.append(eg)

        if on_progress and len(dataset) % 100 == 0:
            on_progress(len(dataset), n)

    if len(dataset) < n:
        import warnings
        warnings.warn(
            f"Only generated {len(dataset)}/{n} unique EGraphs after "
            f"{attempts} attempts. Try relaxing ExprGenConfig (larger depth, "
            "more variables/constants) or set deduplicate=False.",
            RuntimeWarning,
            stacklevel=2,
        )

    return dataset


def _sexpr(node: ExprNode) -> str:
    """Canonical s-expression keyed on integer op IDs (fast, no string lookup)."""
    if not node.children:
        return f"{node.op_id}:{node.param}"
    args = " ".join(_sexpr(c) for c in node.children)
    return f"({node.op_id} {args})"


# ===========================================================================
# Convenience: build OpTable + SymTable from a plain spec, return ExprGenConfig
# ===========================================================================

def build_domain(
    ops:       list[tuple[str, int]],   # e.g. [("add", 2), ("neg", 1)]
    var_names: list[str],               # e.g. ["x", "y", "z"]
    rules:     Optional[list[RewriteRule]] = None,
) -> tuple[OpTable, SymTable, ExprGenConfig, list[RewriteRule]]:
    """
    Wire up OpTable + SymTable and return a matching ExprGenConfig.

    Returns (optable, symtable, expr_gen_config, rules).
    ops and var_names are resolved to integer IDs so ExprGenConfig is
    ready to use without any further string look-ups.
    """
    OT = OpTable()
    for name, arity in ops:
        OT.register(name, arity)

    ST = SymTable()
    var_ids = [ST.intern(v) for v in var_names]

    expr_cfg = ExprGenConfig(
        ops=[(OT.id(name), arity) for name, arity in ops],
        var_ids=var_ids,
    )
    return OT, ST, expr_cfg, rules or []


# ===========================================================================
# Quick smoke-test / usage example
# ===========================================================================

if __name__ == "__main__":
    from EGraph import opat, mvar, cpat, make_math_optable

    # -- Domain setup --------------------------------------------------------
    OT = make_math_optable()
    ST = SymTable()

    # Intern variable names and keep their integer IDs for ExprGenConfig
    var_ids = [ST.intern(v) for v in ["x", "y", "z", "w"]]

    rules = [
        RewriteRule("add-comm",
            lhs=opat(OT, "add", mvar("x"), mvar("y")),
            rhs=opat(OT, "add", mvar("y"), mvar("x"))),
        RewriteRule("mul-comm",
            lhs=opat(OT, "mul", mvar("x"), mvar("y")),
            rhs=opat(OT, "mul", mvar("y"), mvar("x"))),
        RewriteRule("add-zero",
            lhs=opat(OT, "add", mvar("x"), cpat(0)),
            rhs=mvar("x")),
        RewriteRule("mul-one",
            lhs=opat(OT, "mul", mvar("x"), cpat(1)),
            rhs=mvar("x")),
        RewriteRule("mul-zero",
            lhs=opat(OT, "mul", mvar("x"), cpat(0)),
            rhs=cpat(0)),
    ]

    # -- Generator config (all integer IDs, no strings) ----------------------
    expr_cfg = ExprGenConfig(
        ops=[
            (OT.id("add"), 2),
            (OT.id("mul"), 2),
            (OT.id("sub"), 2),
            (OT.id("neg"), 1),
        ],
        var_ids=var_ids,
        const_range=(0, 5),
        min_depth=1,
        max_depth=10,
        leaf_var_prob=0.65,
    )

    dataset_cfg = EGraphDatasetConfig(
        optable=OT,
        symtable=ST,
        expr_cfg=expr_cfg,
        rules=rules,
        apply_rules=False,
        deduplicate=True,
        seed=42,
    )

    # -- Generate ------------------------------------------------------------
    print("Generating 1 000 EGraphs …")
    dataset = generate_egraphs(
        dataset_cfg,
        n=100_000,
        on_progress=lambda cur, tot: print(f"  {cur}/{tot}", end="\r", flush=True),
    )

    print(f"\nDataset size : {len(dataset)}")
    print(f"e-class counts : min={min(g.eclass_count for g in dataset)}"
          f"  max={max(g.eclass_count for g in dataset)}"
          f"  avg={sum(g.eclass_count for g in dataset)/len(dataset):.1f}")
    print(f"e-node  counts : min={min(g.enode_count for g in dataset)}"
          f"  max={max(g.enode_count for g in dataset)}"
          f"  avg={sum(g.enode_count for g in dataset)/len(dataset):.1f}")

    print("\nSample (first 3 graphs, human-readable via OT+ST):")
    for i, eg in enumerate(dataset[:3]):
        print(f"\n--- Graph {i} ---")
        print(eg.pretty(OT, ST))