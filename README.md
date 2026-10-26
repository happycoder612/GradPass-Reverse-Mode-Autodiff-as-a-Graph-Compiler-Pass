# GradPass: Reverse-Mode Autodiff as a Graph Compiler Pass

A minimal tensor compiler pipeline in pure Python/NumPy. Instead of recording operations on a runtime tape (the PyTorch eager approach), GradPass treats automatic differentiation as a **source-to-source compiler pass**: it reads a forward computation graph and *emits a new graph* that computes the gradients. That backward graph can then be optimized like any other program before it runs.

> Educational project. It supports a small set of ops on a NumPy backend and is meant to show the core ideas clearly, not to compete with production frameworks.

## How it works

```
 forward graph (IR)
        │
        ▼
  grad()  ── reverse-mode AD pass ──►  backward graph (new IR nodes)
        │
        ▼
  cse()   ── common-subexpression elimination ──►  optimized backward graph
        │
        ▼
  run()   ── interpreter (NumPy kernels) ──►  numeric gradients
```

1. **IR:** every computation is a `Node(op, inputs)`. Supported ops: `add`, `mul`, `matmul`, `relu`, `transpose`, `sum_all`, plus helper ops (`step`, `sum_to_like`, `broadcast_like`) that the AD pass emits.
2. **Autodiff pass (`grad`):** walks the graph in reverse topological order. Each op has a local gradient rule, and the chain rule composes them. Gradients from multiple uses of the same node are summed, and broadcasting is undone with `sum_to_like`.
3. **Optimization pass (`cse`):** merges structurally identical nodes in the generated backward graph.
4. **Interpreter (`run`):** executes any graph with NumPy kernels.

## Quick start

```bash
pip install numpy pytest
python mini_autodiff.py     # builds a 2-layer MLP, generates gradients, trains it
pytest -q                   # gradient-checking test suite
```

Example output:

```
backward graph nodes: 35 -> 33 after CSE
max gradient error vs finite differences: 6.74e-10
step   0  loss 2249.604
step 100  loss 45.831
step 200  loss 31.582
step 300  loss 26.253
```

## Example

```python
from mini_autodiff import var, matmul, add, relu, mul, sum_all, grad, cse, run

x, W, b = var("x"), var("W"), var("b")
loss = sum_all(relu(add(matmul(x, W), b)))

dW, db = cse(grad(loss, [W, b]))   # compiler pass: forward graph -> backward graph
grads = run([dW, db], {"x": ..., "W": ..., "b": ...})
```

## Correctness

Every gradient rule is verified by **numerical gradient checking**: the compiler-generated gradient is compared with a central finite-difference estimate. `test_grad.py` covers each op, shared-node accumulation (`x*x`), broadcasting, a full 2-layer network, and CSE (results unchanged, graph smaller).

## Project structure

```
mini_autodiff.py   IR, interpreter, autodiff pass, CSE pass, training demo
test_grad.py       gradient-checking and pass-correctness tests
```

## Limitations

- Small op set; no softmax, conv, or indexing yet.
- NumPy interpreter only: no code generation, so no speed-ups are claimed.
- CSE is the only optimization pass so far.
- Gradients are computed for a single scalar loss.

## Roadmap

- [ ] More ops (`sub`, `div`, `exp`, `log`, `softmax`) with gradient rules and tests
- [ ] Elementwise operator fusion pass on the backward graph
- [ ] Rematerialization (recompute cheap activations instead of storing them) and a peak-memory measurement
- [ ] Dead-code elimination and constant folding
- [ ] Emit C++/CUDA kernels for fused groups

## References

- Baydin et al., *Automatic Differentiation in Machine Learning: a Survey* (2018)
- JAX autodiff cookbook (jax.readthedocs.io)
