"""
Mini autodiff-as-a-compiler-pass.

Idea: build the math as a graph (IR). A compiler pass reads the forward graph
and WRITES a new graph that computes the gradients. Then we optimize that
backward graph (here: common-subexpression elimination) and run it.
"""
import numpy as np


# ---------------------------------------------------------------- 1. the IR
class Node:
    def __init__(self, op, inputs=(), value=None, name=None):
        self.op, self.inputs, self.value, self.name = op, tuple(inputs), value, name


def var(name): return Node("var", name=name)
def const(v): return Node("const", value=np.asarray(v, dtype=float))
def add(a, b): return Node("add", (a, b))
def mul(a, b): return Node("mul", (a, b))
def matmul(a, b): return Node("matmul", (a, b))
def relu(a): return Node("relu", (a,))
def step(a): return Node("step", (a,))            # relu'(x): 1 if x>0 else 0
def transpose(a): return Node("transpose", (a,))
def sum_all(a): return Node("sum_all", (a,))
def sum_to_like(g, like): return Node("sum_to_like", (g, like))      # undo broadcasting
def broadcast_like(g, like): return Node("broadcast_like", (g, like))


def topo(outputs):
    """Nodes in dependency order (inputs before the nodes that use them)."""
    order, seen = [], set()

    def visit(n):
        if id(n) in seen:
            return
        seen.add(id(n))
        for i in n.inputs:
            visit(i)
        order.append(n)

    for o in outputs:
        visit(o)
    return order


# ---------------------------------------------------------- 2. interpreter
def _sum_to_like(g, like):
    while g.ndim > like.ndim:
        g = g.sum(axis=0)
    for ax, s in enumerate(like.shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


KERNELS = {
    "add": lambda a, b: a + b,
    "mul": lambda a, b: a * b,
    "matmul": lambda a, b: a @ b,
    "relu": lambda a: np.maximum(a, 0),
    "step": lambda a: (a > 0).astype(float),
    "transpose": lambda a: a.T,
    "sum_all": lambda a: np.array(a.sum()),
    "sum_to_like": _sum_to_like,
    "broadcast_like": lambda g, like: np.broadcast_to(g, like.shape).copy(),
}


def run(outputs, feeds):
    vals = {}
    for n in topo(outputs):
        if n.op == "var":
            vals[id(n)] = feeds[n.name]
        elif n.op == "const":
            vals[id(n)] = n.value
        else:
            vals[id(n)] = KERNELS[n.op](*[vals[id(i)] for i in n.inputs])
    return [vals[id(o)] for o in outputs]


# ------------------------------------------- 3 + 4. the autodiff compiler pass
def grad(loss, wrt):
    """Return NEW graph nodes that compute d(loss)/d(w) for each w in wrt."""
    g = {id(loss): const(1.0)}           # gradient of the loss w.r.t. itself

    def acc(n, contrib):                  # a node used twice gets gradients summed
        g[id(n)] = add(g[id(n)], contrib) if id(n) in g else contrib

    for n in reversed(topo([loss])):      # walk backwards
        if id(n) not in g or not n.inputs:
            continue
        dz = g[id(n)]
        if n.op == "add":
            a, b = n.inputs
            acc(a, sum_to_like(dz, a))
            acc(b, sum_to_like(dz, b))
        elif n.op == "mul":
            a, b = n.inputs
            acc(a, sum_to_like(mul(dz, b), a))
            acc(b, sum_to_like(mul(dz, a), b))
        elif n.op == "matmul":
            a, b = n.inputs
            acc(a, matmul(dz, transpose(b)))
            acc(b, matmul(transpose(a), dz))
        elif n.op == "relu":
            (a,) = n.inputs
            acc(a, mul(dz, step(a)))
        elif n.op == "transpose":
            (a,) = n.inputs
            acc(a, transpose(dz))
        elif n.op == "sum_all":
            (a,) = n.inputs
            acc(a, broadcast_like(dz, a))
        else:
            raise NotImplementedError(f"no gradient rule for {n.op}")
    return [g[id(w)] for w in wrt]


# ------------------------------------------------ 6. optimization pass: CSE
def cse(outputs):
    """Merge nodes that compute the exact same thing."""
    cache, memo = {}, {}

    def rebuild(n):
        if id(n) in memo:
            return memo[id(n)]
        ins = tuple(rebuild(i) for i in n.inputs)
        if n.op == "var":
            key = ("var", n.name)
        elif n.op == "const":
            key = ("const", id(n))
        else:
            key = (n.op, tuple(id(i) for i in ins))
        if key not in cache:
            cache[key] = Node(n.op, ins, n.value, n.name)
        memo[id(n)] = cache[key]
        return cache[key]

    return [rebuild(o) for o in outputs]


# ------------------------------------------------------------------- demo
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Build a tiny 2-layer network + squared-error loss as a graph
    x, y = var("x"), var("y")
    W1, b1, W2, b2 = var("W1"), var("b1"), var("W2"), var("b2")
    h = relu(add(matmul(x, W1), b1))
    pred = add(matmul(h, W2), b2)
    diff = add(pred, mul(const(-1.0), y))
    loss = sum_all(mul(diff, diff))
    params = [W1, b1, W2, b2]

    # The compiler pass writes the backward graph
    grads = grad(loss, params)
    opt_grads = cse(grads)
    print(f"backward graph nodes: {len(topo(grads))} -> {len(topo(opt_grads))} after CSE")

    # 5. Verify with finite differences
    X = rng.normal(size=(8, 3))
    Y = rng.normal(size=(8, 1))
    P = {"W1": rng.normal(size=(3, 4)), "b1": rng.normal(size=(1, 4)),
         "W2": rng.normal(size=(4, 1)), "b2": rng.normal(size=(1, 1))}
    feeds = {"x": X, "y": Y, **P}
    analytic = run(opt_grads, feeds)

    worst = 0.0
    for name, ag in zip(["W1", "b1", "W2", "b2"], analytic):
        num = np.zeros_like(P[name])
        for idx in np.ndindex(*P[name].shape):
            old = P[name][idx]
            P[name][idx] = old + 1e-5
            lp = run([loss], {"x": X, "y": Y, **P})[0]
            P[name][idx] = old - 1e-5
            lm = run([loss], {"x": X, "y": Y, **P})[0]
            P[name][idx] = old
            num[idx] = (lp - lm) / 2e-5
        worst = max(worst, np.abs(num - ag).max())
    print(f"max gradient error vs finite differences: {worst:.2e}")
    assert worst < 1e-5, "gradients are wrong!"

    # Train using the compiler-generated gradients
    Xtr = rng.normal(size=(64, 3))
    Ytr = (Xtr.sum(axis=1, keepdims=True)) ** 2
    P = {"W1": rng.normal(size=(3, 8)) * 0.5, "b1": np.zeros((1, 8)),
         "W2": rng.normal(size=(8, 1)) * 0.5, "b2": np.zeros((1, 1))}
    for step_i in range(301):
        out = run([loss] + opt_grads, {"x": Xtr, "y": Ytr, **P})
        for name, gr in zip(["W1", "b1", "W2", "b2"], out[1:]):
            P[name] -= 0.001 * gr
        if step_i % 100 == 0:
            print(f"step {step_i:3d}  loss {float(out[0]):.3f}")
