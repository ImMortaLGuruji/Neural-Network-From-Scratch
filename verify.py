"""
verify.py
---------
Gradient verification for the backpropagation implementation.

Run this after any change to layers.py, activations.py, or losses.py
to confirm the backward pass is mathematically correct.

Usage
-----
    python verify.py

Method: centred finite differences
------------------------------------
For each weight w_i, the numerical gradient is estimated as:

    dL/dw_i ≈ ( L(w_i + h) - L(w_i - h) ) / (2h)

This is then compared to the analytical gradient computed by backward().
Agreement to ~4 significant figures means the backward pass is correct.

A small network (8 inputs, 6 hidden, 4 outputs) is used so the check
runs in seconds. float64 is required - float32 does not have enough
precision for the finite differences to be accurate.
"""

import numpy as np
from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CrossEntropyLoss


def _forward(
    layers: list, X: np.ndarray, loss_fn: CrossEntropyLoss, Y: np.ndarray
) -> float:
    """Run a full forward pass and return the scalar loss."""
    out = X
    for layer in layers:
        out = layer.forward(out)
    return loss_fn.forward(out, Y)


def _numerical_gradient(
    layers, loss_fn, X, Y, param: np.ndarray, h: float = 1e-5
) -> np.ndarray:
    """
    Estimate the gradient of every element in `param` by finite differences.

    For each element p[i]:
        p[i] += h  →  loss_plus
        p[i] -= h  →  loss_minus
        grad[i] = (loss_plus - loss_minus) / (2h)

    This requires 2 * param.size forward passes, so keep the network small.
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])

    while not it.finished:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + h
        loss_plus = _forward(layers, X, loss_fn, Y)

        param[idx] = orig - h
        loss_minus = _forward(layers, X, loss_fn, Y)

        param[idx] = orig
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        it.iternext()

    return grad


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """
    Scalar relative error between two gradient arrays.

        err = max|a - n| / (max|a| + max|n| + 1e-12)

    Below 1e-4: correct. Above 1e-2: there is a bug.
    """
    num = np.abs(analytical - numerical).max()
    denom = np.abs(analytical).max() + np.abs(numerical).max() + 1e-12
    return float(num / denom)


def run() -> None:
    SEP = "─" * 58

    print("=" * 58)
    print("  Gradient Verification  (centred finite differences)")
    print("=" * 58)

    rng = np.random.default_rng(0)

    # Small network - enough to exercise every path, fast to check
    N, D_in, D_h, D_out = 6, 8, 6, 4

    dense1 = Dense(D_in, D_h)
    relu1 = ReLU()
    dense2 = Dense(D_h, D_out)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()

    # float64 required - float32 is too noisy for finite differences
    for layer in [dense1, dense2]:
        layer.W = layer.W.astype(np.float64)
        layer.b = layer.b.astype(np.float64)

    layers = [dense1, relu1, dense2, softmax]

    X = rng.standard_normal((N, D_in)).astype(np.float64)
    Y = np.eye(D_out)[rng.integers(0, D_out, size=N)]

    # Analytical gradients via backward()
    _forward(layers, X, loss_fn, Y)
    dZ = loss_fn.backward()
    dA = dense2.backward(dZ)
    dZ1 = relu1.backward(dA)
    dense1.backward(dZ1)

    # Compare against numerical estimates
    print(f"\n  {'Parameter':<10}  {'Relative error':<18}  Status")
    print(f"  {SEP}")

    THRESHOLD = 1e-4
    all_passed = True

    checks = [
        ("dense2.W", dense2.W, dense2.dW),
        ("dense2.b", dense2.b, dense2.db),
        ("dense1.W", dense1.W, dense1.dW),
        ("dense1.b", dense1.b, dense1.db),
    ]

    for name, param, grad_analytical in checks:
        grad_numerical = _numerical_gradient(layers, loss_fn, X, Y, param)
        err = _relative_error(grad_analytical, grad_numerical)
        ok = err < THRESHOLD
        all_passed = all_passed and ok
        status = "✓" if ok else "✗  FAIL"
        print(f"  {name:<10}  {err:.2e}                  {status}")

    print(f"  {SEP}")

    if all_passed:
        print(f"\n  All gradients verified. Backpropagation is correct.\n")
    else:
        print(f"\n  FAILED - check backward() in layers.py / activations.py.\n")

    assert all_passed, "Gradient check failed."


if __name__ == "__main__":
    run()
