from pulp import LpMaximize, LpProblem, LpVariable, PULP_CBC_CMD

def solve_lp(m):
    model = LpProblem("LP", LpMaximize)
    x = [LpVariable(f"x{i+1}", lowBound=0) for i in range(4)]
    model += x[0] - 2*x[1] + 3*x[2] + m*x[3]
    model += 2*x[0] + x[2] - x[3] == 6
    model += x[1] + 2*x[2] - 3*x[3] == 3
    model += 2*x[0] - x[1] + x[3] == 5
    model.solve(PULP_CBC_CMD(msg=0))
    print(f"m = {m}")
    if model.status == 1:
        print(f"f = {model.objective.value()}")
        print([xi.varValue for xi in x])
    elif model.status == 2:
        print("Khong co nghiem.")
    else:
        print("Khong bi chan duoi.")

for m in [2, -1, -3]:
    solve_lp(m)
