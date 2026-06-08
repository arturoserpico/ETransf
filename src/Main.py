import Policy
from EGraph import *

OT = make_math_optable()
ST = SymTable()

eg = EGraph()

a   = eg.add(make_var(OT, ST, "a"))
two = eg.add(make_const(3))
mul = eg.add(make_op(OT, "mul", a, two))
div = eg.add(make_op(OT, "div", mul, two))
add = eg.add(make_op(OT, "add", mul, div))

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

print(eg.extract(add))

print(eg.pretty())

print(eg.is_rule_applicable_at(rules[2], eg._classes[a].nodes[0]))


model = Policy.Policy(rules, 128, 3, 2, 2, 8, 4, 4, 4, len(OT))

out = model(eg)

print(out.shape)
print(out)