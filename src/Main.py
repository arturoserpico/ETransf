import Policy
import EGraph

OT = EGraph.make_math_optable()
ST = EGraph.SymTable()

eg = EGraph.EGraph()

a   = eg.add(EGraph.make_var(OT, ST, "a"))
two = eg.add(EGraph.make_const(2))
mul = eg.add(EGraph.make_op(OT, "mul", a, two))
div = eg.add(EGraph.make_op(OT, "div", mul, two))
add = eg.add(EGraph.make_op(OT, "add", mul, div))

print(eg.pretty())

model = Policy.Policy(128, 3, 2, 2, 8, 4, 4, 4, len(OT), 10)

out = model(eg)

print(out.shape)
print(out)