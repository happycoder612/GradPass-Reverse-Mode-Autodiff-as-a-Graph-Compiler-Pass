"""
Tests for mini_autodiff.

Core idea: gradient checking. For each op, compare the gradient produced by
our compiler pass against a numerical estimate from finite differences:

    dL/dw  ~=  (L(w + eps) - L(w - eps)) / (2 * eps)

If the two agree, the gradient rules (and the graph walk) are correct.
Run with:  pytest -q
"""
import numpy as np
import pytest

from mini_autodiff import (
    var, const, add, mul, matmul, relu, sum_all, exp, softmax,
    grad, cse, run, run_mem, topo,
)

RNG = np.random.default_rng(42)


def numeric_grad(loss, feeds, name, eps=1e-5):
    """Finite-difference gradient of `loss` w.r.t. the input called `name`."""
    x = feeds[name]
    out = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        old = x[idx]
        x[idx] = old + eps
        plus = run([loss], feeds)[0]
        x[idx] = old - eps
        minus = run([loss], feeds)[0]
        x[idx] = old
        out[idx] = (plus - minus) / (2 * eps)
    return out


def check(loss, params, feeds, use_cse=False, tol=1e-6):
    grads = grad(loss, params)
    if use_cse:
        grads = cse(grads)
    analytic = run(grads, feeds)
    for p, a in zip(params, analytic):
        n = numeric_grad(loss, feeds, p.name)
        assert np.allclose(a, n, atol=tol), f"gradient mismatch for {p.name}"


def test_add_and_mul():
    x, y = var("x"), var("y")
    loss = sum_all(mul(add(x, y), x))
    feeds = {"x": RNG.normal(size=(3, 2)), "y": RNG.normal(size=(3, 2))}
    check(loss, [x, y], feeds)


def test_matmul():
    A, B = var("A"), var("B")
    loss = sum_all(mul(matmul(A, B), matmul(A, B)))
    feeds = {"A": RNG.normal(size=(4, 3)), "B": RNG.normal(size=(3, 5))}
    check(loss, [A, B], feeds)


def test_relu():
    x = var("x")
    loss = sum_all(mul(relu(x), relu(x)))
    # keep values away from 0, where relu is not differentiable
    vals = RNG.normal(size=(4, 4))
    vals[np.abs(vals) < 0.1] += 0.5
    check(loss, [x], {"x": vals})


def test_exp():
    x = var("x")
    loss = sum_all(exp(x))
    vals = RNG.normal(size=(3, 4))
    check(loss, [x], {"x": vals})
    (g,) = run(grad(loss, [x]), {"x": vals})
    assert np.allclose(g, np.exp(vals))            # d/dx exp(x) = exp(x)


def test_softmax_forward_rows_sum_to_one():
    x = var("x")
    (out,) = run([softmax(x)], {"x": RNG.normal(size=(5, 7)) * 50})   # large values: stability check
    assert np.allclose(out.sum(axis=-1), 1.0)
    assert np.isfinite(out).all()


def test_softmax_gradient():
    # weight by w so the loss is not constant (rows of softmax always sum to 1)
    x, w = var("x"), var("w")
    loss = sum_all(mul(softmax(x), w))
    feeds = {"x": RNG.normal(size=(4, 5)), "w": RNG.normal(size=(4, 5))}
    check(loss, [x, w], feeds)


def test_softmax_and_exp_compose():
    # exp(softmax(x)) checks that the two new gradient rules chain correctly
    x, t = var("x"), var("t")
    loss = sum_all(mul(exp(softmax(x)), t))
    feeds = {"x": RNG.normal(size=(3, 6)), "t": RNG.normal(size=(3, 6))}
    check(loss, [x, t], feeds)


def test_shared_node_gradients_accumulate():
    """x is used twice in x*x, so dL/dx must be 2x (both paths summed)."""
    x = var("x")
    loss = sum_all(mul(x, x))
    vals = RNG.normal(size=(3, 3))
    (g,) = run(grad(loss, [x]), {"x": vals})
    assert np.allclose(g, 2 * vals)


def test_broadcast_bias_gradient_shape():
    """A (1, n) bias added to a (batch, n) matrix needs its gradient summed back."""
    x, b = var("x"), var("b")
    loss = sum_all(mul(add(x, b), add(x, b)))
    feeds = {"x": RNG.normal(size=(6, 4)), "b": RNG.normal(size=(1, 4))}
    (gb,) = run(grad(loss, [b]), feeds)
    assert gb.shape == (1, 4)
    check(loss, [x, b], feeds)


def test_two_layer_network():
    x, y = var("x"), var("y")
    W1, b1, W2, b2 = var("W1"), var("b1"), var("W2"), var("b2")
    h = relu(add(matmul(x, W1), b1))
    pred = add(matmul(h, W2), b2)
    diff = add(pred, mul(const(-1.0), y))
    loss = sum_all(mul(diff, diff))
    feeds = {
        "x": RNG.normal(size=(8, 3)), "y": RNG.normal(size=(8, 1)),
        "W1": RNG.normal(size=(3, 4)), "b1": RNG.normal(size=(1, 4)),
        "W2": RNG.normal(size=(4, 1)), "b2": RNG.normal(size=(1, 1)),
    }
    check(loss, [W1, b1, W2, b2], feeds)


def test_cse_keeps_results_and_shrinks_graph():
    x, y = var("x"), var("y")
    loss = sum_all(mul(add(x, y), add(x, y)))    # add(x, y) is computed twice
    grads = grad(loss, [x, y])
    opt = cse(grads)
    feeds = {"x": RNG.normal(size=(3, 3)), "y": RNG.normal(size=(3, 3))}
    for a, b in zip(run(grads, feeds), run(opt, feeds)):
        assert np.allclose(a, b)
    assert len(topo(opt)) < len(topo(grads))


def _mlp_graph():
    x, y = var("x"), var("y")
    W1, b1, W2, b2 = var("W1"), var("b1"), var("W2"), var("b2")
    h = relu(add(matmul(x, W1), b1))
    pred = add(matmul(h, W2), b2)
    diff = add(pred, mul(const(-1.0), y))
    loss = sum_all(mul(diff, diff))
    feeds = {
        "x": RNG.normal(size=(32, 8)), "y": RNG.normal(size=(32, 1)),
        "W1": RNG.normal(size=(8, 16)), "b1": RNG.normal(size=(1, 16)),
        "W2": RNG.normal(size=(16, 1)), "b2": RNG.normal(size=(1, 1)),
    }
    return loss, cse(grad(loss, [W1, b1, W2, b2])), feeds


def test_run_mem_matches_run():
    loss, grads, feeds = _mlp_graph()
    expected = run([loss] + grads, feeds)
    for free in (True, False):
        got, _ = run_mem([loss] + grads, feeds, free=free)
        for e, g in zip(expected, got):
            assert np.allclose(e, g)


def test_freeing_lowers_peak_memory():
    loss, grads, feeds = _mlp_graph()
    _, keep_all = run_mem([loss] + grads, feeds, free=False)
    _, freed = run_mem([loss] + grads, feeds, free=True)
    assert freed < keep_all


def test_freeing_chain_peak_is_small():
    """A long elementwise chain only ever needs ~2 live arrays when freeing."""
    x = var("x")
    node = x
    for _ in range(10):
        node = relu(node)
    vals = RNG.normal(size=(100, 100))
    _, keep_all = run_mem([node], {"x": vals}, free=False)
    _, freed = run_mem([node], {"x": vals}, free=True)
    assert keep_all == 10 * vals.nbytes
    assert freed == 2 * vals.nbytes


def test_unsupported_op_raises():
    from mini_autodiff import Node
    x = var("x")
    loss = Node("sum_all", (Node("mystery_op", (x,)),))
    with pytest.raises(NotImplementedError):
        grad(loss, [x])
