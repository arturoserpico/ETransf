from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from collections import defaultdict
import itertools
 
 
# ===========================================================================
# Operator Table
# ===========================================================================
 
class OpTable:
    """
    Bidirectional map  operator-name <-> integer-ID, with arity metadata.
 
    Two IDs are permanently reserved:
      0  →  "var"    leaf: ENode.param = symbol ID in a SymTable
      1  →  "const"  leaf: ENode.param = the literal integer value
 
    All other operators are registered by the caller via register().
    """
 
    OP_VAR   = 0
    OP_CONST = 1
 
    def __init__(self) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._arity:      dict[int, int] = {}   # op_id -> arity (-1 = variadic)
        self._counter = itertools.count(2)       # 0 and 1 are reserved
 
        for name, oid, arity in [("var", 0, 0), ("const", 1, 0)]:
            self._name_to_id[name] = oid
            self._id_to_name[oid]  = name
            self._arity[oid]       = arity
 
    def register(self, name: str, arity: int = -1) -> int:
        """Register *name* as an operator and return its integer ID."""
        if name in self._name_to_id:
            return self._name_to_id[name]
        oid = next(self._counter)
        self._name_to_id[name] = oid
        self._id_to_name[oid]  = name
        self._arity[oid]       = arity
        return oid
 
    def id(self, name: str) -> int:
        """Return the ID for an already-registered operator name."""
        try:
            return self._name_to_id[name]
        except KeyError:
            raise KeyError(f"Unknown operator {name!r}. Call register() first.") from None
 
    def name(self, oid: int) -> str:
        """Return the human-readable name for integer operator ID *oid*."""
        return self._id_to_name.get(oid, f"<op#{oid}>")
 
    def arity(self, oid: int) -> int:
        return self._arity.get(oid, -1)
 
    def __repr__(self) -> str:
        rows = [
            f"  {oid:>3}  {self._id_to_name[oid]!r:<14} arity={self._arity[oid]}"
            for oid in sorted(self._id_to_name)
        ]
        return "OpTable:\n" + "\n".join(rows)
    
    def __len__(self) -> int:
        return len(self._name_to_id)

 
 
def make_math_optable() -> OpTable:
    """Return an OpTable pre-loaded with standard arithmetic operators."""
    t = OpTable()
    for name, arity in [
        ("add", 2), ("sub", 2), ("mul", 2), ("div", 2),
        ("neg", 1), ("pow", 2),
    ]:
        t.register(name, arity)
    return t
 
 
# ===========================================================================
# Symbol Table
# ===========================================================================
 
class SymTable:
    """
    Bidirectional map  symbol-name <-> integer-ID.
 
    Symbol IDs are stored in ENode.param when op == OpTable.OP_VAR,
    letting leaf nodes represent named variables without string storage.
    """
 
    def __init__(self) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_name: dict[int, str] = {}
        self._counter = itertools.count()
 
    def intern(self, name: str) -> int:
        """Return the ID for *name*, registering it if new."""
        if name not in self._name_to_id:
            sid = next(self._counter)
            self._name_to_id[name] = sid
            self._id_to_name[sid]  = name
        return self._name_to_id[name]
 
    def name(self, sid: int) -> str:
        return self._id_to_name.get(sid, f"<sym#{sid}>")
 
    def __repr__(self) -> str:
        rows = [f"  {sid:>3}  {self._id_to_name[sid]!r}"
                for sid in sorted(self._id_to_name)]
        return "SymTable:\n" + "\n".join(rows)
 
 
# ===========================================================================
# E-Node
# ===========================================================================
 
@dataclass(frozen=True)
class ENode:
    """
    An e-node: an integer-encoded operator applied to zero or more e-class IDs.
 
    Fields
    ------
    op       : int             – operator ID looked up in OpTable
    children : tuple[int, ...] – e-class IDs of children (empty for leaves)
    param    : int             – leaf parameter (0 for interior / unused):
                 op == OP_VAR   →  symbol ID in SymTable
                 op == OP_CONST →  literal integer value
                 any other leaf →  caller-defined extra data
    """
 
    op:       int
    children: tuple[int, ...]
    param:    int = 0
 
    def __repr__(self) -> str:
        """Compact representation using raw integer codes."""
        if not self.children:
            return f"ENode(op={self.op}, param={self.param})"
        return f"ENode(op={self.op}, children={self.children})"
 
    def format(
        self,
        optable:  OpTable,
        symtable: Optional[SymTable] = None,
    ) -> str:
        """Human-readable string, names resolved through *optable* / *symtable*."""
        op_name = optable.name(self.op)
        if not self.children:
            if self.op == OpTable.OP_VAR:
                return symtable.name(self.param) if symtable else str(self.param)
            if self.op == OpTable.OP_CONST:
                return str(self.param)
            return op_name
        args = " ".join(str(c) for c in self.children)
        return f"({op_name} {args})"

    def child_classes(self, egraph: EGraph) -> list[EClass]:
        """
        Return the canonical EClass for each child of this node, in tuple order.
        """
        return [egraph._classes[egraph.find(cid)] for cid in self.children]
 
 
# ---------------------------------------------------------------------------
# ENode builder helpers
# ---------------------------------------------------------------------------
 
def make_var(optable: OpTable, symtable: SymTable, name: str) -> ENode:
    """Build a variable leaf ENode."""
    return ENode(op=OpTable.OP_VAR, children=(), param=symtable.intern(name))
 
def make_const(value: int) -> ENode:
    """Build an integer constant leaf ENode."""
    return ENode(op=OpTable.OP_CONST, children=(), param=value)
 
def make_op(optable: OpTable, op_name: str, *child_class_ids: int) -> ENode:
    """Build an interior ENode from a registered operator name and e-class IDs."""
    return ENode(op=optable.id(op_name), children=tuple(child_class_ids))
 
 
# ===========================================================================
# Union-Find
# ===========================================================================
 
class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}
        self._rank:   dict[int, int] = {}
 
    def make(self, x: int) -> int:
        self._parent[x] = x
        self._rank[x]   = 0
        return x
 
    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]   # path halving
            x = self._parent[x]
        return x
 
    def union(self, x: int, y: int) -> int:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1
        return rx
 
 
# ===========================================================================
# E-Class
# ===========================================================================
 
@dataclass
class EClass:
    id:      int
    nodes:   list[ENode]             = field(default_factory=list)
    parents: list[tuple[ENode,int]] = field(default_factory=list)
    # parents = (enode, eclass_id) pairs where this class appears as a child
    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EClass) and self.id == other.id
 
 
# ===========================================================================
# E-Graph
# ===========================================================================
 
class EGraph:
    def __init__(self) -> None:
        self._uf:        UnionFind              = UnionFind()
        self._classes:   dict[int, EClass]      = {}
        self._memo:      dict[ENode, int]       = {}   # enode -> canonical eclass id
        self._worklist:  list[tuple[int, int]]  = []   # pending merges
        self._id_counter = itertools.count()
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _new_class(self) -> EClass:
        cid = next(self._id_counter)
        self._uf.make(cid)
        cls = EClass(id=cid)
        self._classes[cid] = cls
        return cls
 
    def find(self, cls_id: int) -> int:
        return self._uf.find(cls_id)
 
    def _canonicalize(self, node: ENode) -> ENode:
        """Return a copy of *node* with each child replaced by its canonical ID."""
        return ENode(node.op, tuple(self.find(c) for c in node.children), node.param)
 
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
 
    def add(self, node: ENode) -> int:
        """Add an e-node; return the e-class ID it belongs to."""
        node = self._canonicalize(node)
        if node in self._memo:
            return self.find(self._memo[node])
 
        cls = self._new_class()
        cls.nodes.append(node)
        self._memo[node] = cls.id
 
        for child_id in node.children:
            self._classes[self.find(child_id)].parents.append((node, cls.id))
 
        return cls.id
 
    def merge(self, id1: int, id2: int) -> int:
        """Assert that two e-classes are equal and schedule a rebuild."""
        id1, id2 = self.find(id1), self.find(id2)
        if id1 == id2:
            return id1
 
        merged = self._uf.union(id1, id2)
        loser  = id2 if merged == id1 else id1
 
        winner_cls = self._classes[merged]
        loser_cls  = self._classes[loser]
        winner_cls.nodes  |= loser_cls.nodes
        winner_cls.parents += loser_cls.parents
 
        self._worklist.append((merged, loser))
        return merged
 
    def rebuild(self) -> None:
        """
        Restore the congruence invariant: e-nodes that are equal up to
        canonical child IDs must be in the same e-class.
        """
        while self._worklist:
            merged, _ = self._worklist.pop()
            cls = self._classes[self.find(merged)]
            to_reprocess, cls.parents = cls.parents, []
 
            for (node, node_cls_id) in to_reprocess:
                canon        = self._canonicalize(node)
                node_cls_id  = self.find(node_cls_id)
 
                if canon in self._memo:
                    existing = self.find(self._memo[canon])
                    if existing != node_cls_id:
                        self.merge(existing, node_cls_id)
                else:
                    self._memo[canon] = node_cls_id
                    for child_id in canon.children:
                        self._classes[self.find(child_id)].parents.append(
                            (canon, self.find(node_cls_id))
                        )
 
            stale = [k for k, v in self._memo.items() if self.find(v) != v]
            for k in stale:
                self._memo[k] = self.find(self._memo[k])
 
    # ------------------------------------------------------------------
    # Rewriting
    # ------------------------------------------------------------------
 
    def apply_rules(self, rules: list[RewriteRule]) -> int:
        """Apply all rules repeatedly until saturation. Returns iteration count."""
        for iteration in range(1, 1001):
            matches: list[tuple[RewriteRule, dict[str, int]]] = []
            for rule in rules:
                for cls_id in list(self._classes):
                    cls_id = self.find(cls_id)
                    for node in self._classes[cls_id].nodes:
                        env = rule.lhs.match(node, cls_id, self)
                        if env is not None:
                            matches.append((rule, env))
 
            if not matches:
                return iteration
 
            for rule, env in matches:
                new_id = rule.rhs.apply(env, self)
                src_id = env.get("__root__")
                if src_id is not None:
                    self.merge(src_id, new_id)
 
            self.rebuild()
        return 1000   # did not saturate
 
    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
 
    def extract(
        self,
        cls_id:   int,
        optable:  Optional[OpTable]  = None,
        symtable: Optional[SymTable] = None,
        cost_fn:  Optional[Callable[[ENode, list[float]], float]] = None,
    ) -> tuple[str, float]:
        """
        Extract the cheapest expression from *cls_id*.
        Returns (pretty_string, cost).  Default cost = AST node count.
        """
        if cost_fn is None:
            cost_fn = lambda node, child_costs: 1 + sum(child_costs)
 
        best: dict[int, tuple[Optional[str], float]] = {}
 
        def visit(cid: int) -> tuple[Optional[str], float]:
            cid = self.find(cid)
            if cid in best:
                return best[cid]
            best[cid] = (None, float("inf"))    # sentinel against cycles
 
            for node in self._classes[cid].nodes:
                child_results = [visit(c) for c in node.children]
                if any(expr is None for expr, _ in child_results):
                    continue
                child_costs  = [cost for _, cost in child_results]
                child_exprs  = [expr for expr, _ in child_results]
                c = cost_fn(node, child_costs)
                if c < best[cid][1]:
                    if optable:
                        op_name = optable.name(node.op)
                        if not node.children:
                            if node.op == OpTable.OP_VAR:
                                s = symtable.name(node.param) if symtable else str(node.param)
                            elif node.op == OpTable.OP_CONST:
                                s = str(node.param)
                            else:
                                s = op_name
                        else:
                            s = f"({op_name} {' '.join(child_exprs)})"
                    else:
                        # Fall back to raw integer representation
                        if not node.children:
                            s = f"op{node.op}[{node.param}]"
                        else:
                            s = f"(op{node.op} {' '.join(child_exprs)})"
                    best[cid] = (s, c)
 
            return best[cid]
 
        return visit(cls_id)
 
    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
 
    def pretty(
        self,
        optable:  Optional[OpTable]  = None,
        symtable: Optional[SymTable] = None,
    ) -> str:
        groups: dict[int, set[int]] = defaultdict(set)
        for cid in self._classes:
            groups[self.find(cid)].add(cid)
 
        lines = ["EGraph:"]
        for root, members in sorted(groups.items()):
            cls = self._classes[root]
            lines.append(f"  class {root} (ids={sorted(members)}):")
            for node in cls.nodes:
                s = node.format(optable, symtable) if optable else repr(node)
                lines.append(f"    {s}")
        return "\n".join(lines)
 
    def __repr__(self) -> str:
        return self.pretty()
    
    @property
    def eclasses(self) -> dict[int, EClass]:
        """Return only canonical e-classes (one per equivalence group)."""
        return {
            cid: cls
            for cid, cls in self._classes.items()
            if self.find(cid) == cid
        }
    
    @property
    def enodes(self) -> dict[ENode, int]:
        """Return all interned e-nodes mapped to their canonical e-class ID."""
        return {node: self.find(cid) for node, cid in self._memo.items()}
    

    def apply_rule_at(self, rule: RewriteRule, node: ENode) -> Optional[int]:
        """
        Try to apply *rule* at a specific e-node.

        Returns the canonical e-class ID of the new term on success,
        or None if the node isn't in the graph or the LHS doesn't match.
        After a successful match the graph is rebuilt automatically.
        """
        canon = self._canonicalize(node)
        if canon not in self._memo:
            return None

        cls_id = self.find(self._memo[canon])
        env    = rule.lhs.match(canon, cls_id, self)
        if env is None:
            return None

        new_id = rule.rhs.apply(env, self)
        src_id = env.get("__root__")
        if src_id is not None:
            self.merge(src_id, new_id)
        self.rebuild()
        return self.find(new_id)


    def apply_rule_at_class(self, rule: RewriteRule, cls_id: int) -> list[int]:
        """
        Try to apply *rule* at every e-node in e-class *cls_id*.

        Returns a list of new canonical e-class IDs for each successful match
        (may be empty if no node matched).  The graph is rebuilt once at the end.
        """
        cls_id  = self.find(cls_id)
        results = []

        for node in list(self._classes[cls_id].nodes):
            env = rule.lhs.match(node, cls_id, self)
            if env is None:
                continue
            new_id = rule.rhs.apply(env, self)
            src_id = env.get("__root__")
            if src_id is not None:
                self.merge(src_id, new_id)
            results.append(new_id)

        if results:
            self.rebuild()

        return [self.find(r) for r in results]
    
    def is_rule_applicable_at(self, rule: RewriteRule, node: ENode) -> bool:
        """
        Return True if *rule*'s LHS matches *node* in this graph, False otherwise.
        The graph is never modified.
        """
        canon = self._canonicalize(node)
        if canon not in self._memo:
            return False
        cls_id = self.find(self._memo[canon])
        return rule.lhs.match(canon, cls_id, self) is not None

    @property
    def eclass_count(self) -> int:
        """Number of canonical (non-merged) e-classes."""
        return sum(1 for cid in self._classes if self.find(cid) == cid)

    @property
    def enode_count(self) -> int:
        """Number of canonical e-nodes (excludes stale memo entries)."""
        return sum(1 for cid in self._memo.values() if self.find(cid) == cid)
 
 
# ===========================================================================
# Pattern Language for Rewrite Rules
# ===========================================================================
 
@dataclass
class Pattern:
    """
    Pattern node for matching / constructing e-nodes.
 
    Metavariable  (var != None):
        Matches any e-class; binds its ID to the variable name in the env.
 
    Structural pattern (var is None):
        op       : int            – operator ID to match
        children : list[Pattern]  – sub-patterns for each child
        param    : Optional[int]  – if set, also require ENode.param == this value
                                    (used to match a specific constant or variable)
    """
    op:       Optional[int]       = None
    children: list["Pattern"]     = field(default_factory=list)
    var:      Optional[str]       = None
    param:    Optional[int]       = None   # exact param constraint for leaf patterns
 
    # ------------------------------------------------------------------ matching
 
    def match(
        self,
        node:   ENode,
        cls_id: int,
        egraph: EGraph,
        env:    Optional[dict[str, int]] = None,
    ) -> Optional[dict[str, int]]:
        """
        Try to match *node* (which lives in *cls_id*) against this pattern.
        Returns a filled binding environment on success, None on failure.
        """
        if env is None:
            env = {"__root__": cls_id}
 
        # -- metavariable --
        if self.var is not None:
            if self.var in env and egraph.find(env[self.var]) != egraph.find(cls_id):
                return None
            env[self.var] = cls_id
            return env
 
        # -- structural --
        if self.op != node.op or len(self.children) != len(node.children):
            return None
        if self.param is not None and self.param != node.param:
            return None
 
        for pat_child, node_child_id in zip(self.children, node.children):
            canon_child = egraph.find(node_child_id)
 
            if pat_child.var is not None:
                # Metavariable child: bind directly to the child class
                if pat_child.var in env and egraph.find(env[pat_child.var]) != canon_child:
                    return None
                env[pat_child.var] = canon_child
            else:
                # Structural child: try every e-node in the child class
                matched = False
                for child_node in egraph._classes[canon_child].nodes:
                    result = pat_child.match(child_node, canon_child, egraph, dict(env))
                    if result is not None:
                        env     = result
                        matched = True
                        break
                if not matched:
                    return None
 
        return env
 
    # ------------------------------------------------------------------ application
 
    def apply(self, env: dict[str, int], egraph: EGraph) -> int:
        """Build and add an e-node from this pattern, substituting *env* for variables."""
        if self.var is not None:
            return env[self.var]
        child_ids = [c.apply(env, egraph) for c in self.children]
        param     = self.param if self.param is not None else 0
        return egraph.add(ENode(self.op, tuple(child_ids), param))
 
 
@dataclass
class RewriteRule:
    name: str
    lhs:  Pattern
    rhs:  Pattern
 
    def __repr__(self) -> str:
        return f"RewriteRule({self.name!r})"
 
 
# ---------------------------------------------------------------------------
# Pattern builder helpers  (accept string names, resolve via optable)
# ---------------------------------------------------------------------------
 
def opat(optable: OpTable, op_name: str, *children: Pattern) -> Pattern:
    """Structural pattern: match operator *op_name* with sub-patterns *children*."""
    return Pattern(op=optable.id(op_name), children=list(children))
 
def cpat(value: int) -> Pattern:
    """Leaf pattern: match a specific integer constant."""
    return Pattern(op=OpTable.OP_CONST, param=value)
 
def vpat(symtable: SymTable, sym_name: str) -> Pattern:
    """Leaf pattern: match a specific named variable."""
    return Pattern(op=OpTable.OP_VAR, param=symtable.intern(sym_name))
 
def mvar(name: str) -> Pattern:
    """Metavariable pattern: matches any e-class and binds it to *name*."""
    return Pattern(var=name)
 
 
# ===========================================================================
# Demo
# ===========================================================================
 
if __name__ == "__main__":
    OT = make_math_optable()
    ST = SymTable()
 
    print(OT)
    print()
 
    # -----------------------------------------------------------------------
    # Example 1: (a * 2) / 2  →  a
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Example 1: Simplify  (a * 2) / 2  →  a")
    print("=" * 60)
 
    eg = EGraph()
 
    a   = eg.add(make_var(OT, ST, "a"))
    two = eg.add(make_const(2))
    mul = eg.add(make_op(OT, "mul", a, two))
    div = eg.add(make_op(OT, "div", mul, two))
 
    print("Initial e-graph:")
    print(eg.pretty(OT, ST))
    print(f"\nRoot class: {div}  raw repr: {eg._classes[div].nodes}\n")
 
    rules = [
        RewriteRule(
            "cancel-mul-div",
            lhs=opat(OT, "div", opat(OT, "mul", mvar("x"), mvar("y")), mvar("y")),
            rhs=mvar("x"),
        ),
        RewriteRule(
            "mul-one",
            lhs=opat(OT, "mul", mvar("x"), cpat(1)),
            rhs=mvar("x"),
        ),
        RewriteRule(
            "div-one",
            lhs=opat(OT, "div", mvar("x"), cpat(1)),
            rhs=mvar("x"),
        ),
    ]
 
    iters = eg.apply_rules(rules)
    print(f"Saturated in {iters} iteration(s).")
    print("\nFinal e-graph:")
    print(eg.pretty(OT, ST))
 
    result, cost = eg.extract(div, OT, ST)
    print(f"\nExtracted: {result}  (cost={cost})")
 
    # -----------------------------------------------------------------------
    # Example 2: commutativity — add(a, b) ≡ add(b, a)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: commutativity  add(a, b)  ≡  add(b, a)")
    print("=" * 60)
 
    eg2 = EGraph()
    a2  = eg2.add(make_var(OT, ST, "a"))
    b2  = eg2.add(make_var(OT, ST, "b"))
    ab  = eg2.add(make_op(OT, "add", a2, b2))
    ba  = eg2.add(make_op(OT, "add", b2, a2))
    print(f"add(a,b) class={ab},  add(b,a) class={ba}  same={eg2.find(ab)==eg2.find(ba)}")
 
    comm = RewriteRule(
        "add-commute",
        lhs=opat(OT, "add", mvar("x"), mvar("y")),
        rhs=opat(OT, "add", mvar("y"), mvar("x")),
    )
    eg2.apply_rules([comm])
    print(f"After commutativity rule — same? {eg2.find(ab) == eg2.find(ba)}")
 
    # -----------------------------------------------------------------------
    # Example 3: inspect raw integer encoding
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: raw integer encoding of e-nodes")
    print("=" * 60)
 
    print(f"\nOpTable IDs:  var={OT.OP_VAR}  const={OT.OP_CONST}  "
          f"add={OT.id('add')}  mul={OT.id('mul')}  div={OT.id('div')}")
    print(ST)
 
    # Show raw ENodes from example 1
    eg3 = EGraph()
    a3   = eg3.add(make_var(OT, ST, "x"))
    c3   = eg3.add(make_const(42))
    add3 = eg3.add(make_op(OT, "add", a3, c3))
 
    for cls_id, cls in sorted(eg3._classes.items()):
        for node in cls.nodes:
            print(f"  class {cls_id}: {node!r:40s}  →  {node.format(OT, ST)}")
 
    print("\nAll done.")