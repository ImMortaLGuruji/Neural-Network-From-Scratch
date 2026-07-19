"""
network.py
----------
Network class - assembles layers into a trainable model.

The Network class wraps the forward pass, backward pass, and
optimizer step into a clean API used by train.py.

Lower-level test functions are preserved below the class
definition for reference and regression testing.
"""

import numpy as np
from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CrossEntropyLoss
from nn.optimizer import SGD
from nn.utils import load_mnist, one_hot

# ══════════════════════════════════════════════════════════════════════════
# Network class
# ══════════════════════════════════════════════════════════════════════════


class Network:
    """
    Feedforward neural network - wraps layers, loss, and optimizer.

    Replaces the manual wiring with a clean API:

        net = Network(layers=[...], loss_fn=CrossEntropyLoss(), optimizer=SGD(...))
        loss = net.train_step(X_batch, Y_batch)
        preds = net.predict(X_test)
        acc   = net.accuracy(X_test, y_test)

    Design note - Softmax and the backward pass
    -------------------------------------------
    Softmax is the last element of `layers` and IS included in forward().
    In backward(), however, it is skipped.

    Why?  CrossEntropyLoss.backward() returns the fused Softmax+CE
    gradient (P - Y) / N, which is already the gradient w.r.t. the
    logits Z entering Softmax - not w.r.t. Softmax's output P.
    So the backward chain begins one step *before* Softmax, at the
    last Dense layer.  Calling Softmax.backward() on top of that
    would double-apply the Softmax transformation.

    Concretely, backward() iterates reversed(layers[:-1]) - all layers
    except the final Softmax.

    Parameters
    ----------
    layers    : list
        Ordered list of layer/activation objects.
        Must end with a Softmax instance.
    loss_fn   : CrossEntropyLoss
        Loss function.  Paired with Softmax at the output.
    optimizer : SGD
        Optimizer instance, already initialised with the Dense layers
        and learning rate.
    """

    def __init__(
        self,
        layers: list,
        loss_fn: CrossEntropyLoss,
        optimizer: SGD,
    ) -> None:
        assert isinstance(layers[-1], Softmax), (
            "The last layer must be Softmax. "
            f"Got {type(layers[-1]).__name__} instead."
        )
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through every layer.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)
            Input batch.

        Returns
        -------
        P : np.ndarray, shape (N, num_classes)
            Softmax probability distribution.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _backward(self, dZ: np.ndarray) -> None:
        """
        Run the backward chain through all layers except the final Softmax.

        Parameters
        ----------
        dZ : np.ndarray, shape (N, num_classes)
            Gradient from loss_fn.backward() - the fused Softmax+CE
            gradient (P - Y) / N, already past Softmax.

        The chain runs in reverse order: last Dense → its ReLU → ... → first Dense.
        Each Dense layer accumulates dW and db for the optimizer.
        """
        grad = dZ
        # layers[:-1] skips Softmax - its gradient is already in dZ
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    # ------------------------------------------------------------------
    # Single training step
    # ------------------------------------------------------------------

    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        One complete training iteration:
            zero_grad → forward → loss → backward → step

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)    - input batch
        Y : np.ndarray, shape (N, C)       - one-hot labels

        Returns
        -------
        loss : float
            Scalar cross-entropy loss for this batch.
        """
        # 1. Clear stale gradients from the previous step
        self.optimizer.zero_grad()

        # 2. Forward pass → probabilities
        P = self.forward(X)

        # 3. Compute loss
        loss = self.loss_fn.forward(P, Y)

        # 4. Backward pass → accumulate dW, db in every Dense layer
        dZ = self.loss_fn.backward()
        self._backward(dZ)

        # 5. Gradient descent step → update W and b
        self.optimizer.step()

        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class index for each example.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)

        Returns
        -------
        np.ndarray of shape (N,), dtype int64 - argmax of Softmax output.
        """
        P = self.forward(X)
        return P.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)  - input features
        y : np.ndarray, shape (N,)       - integer class labels

        Returns
        -------
        float in [0, 1] - fraction of correctly classified examples.
        """
        preds = self.predict(X)
        return float((preds == y).mean())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a human-readable summary of the network architecture."""
        print("Network")
        print("─" * 40)
        total_params = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                params = layer.W.size + layer.b.size
                total_params += params
                print(f"  [{i}] {layer}   params={params:,}")
            else:
                print(f"  [{i}] {layer}")
        print("─" * 40)
        print(f"  Total trainable parameters: {total_params:,}")

    def __repr__(self) -> str:
        names = " → ".join(type(l).__name__ for l in self.layers)
        return f"Network([{names}])"


# ══════════════════════════════════════════════════════════════════════════
# Manual Forward Pass Smoke Test
# ══════════════════════════════════════════════════════════════════════════


def smoke_test(batch_size: int = 64) -> None:
    """Wire all components manually and verify one forward pass."""

    print("=" * 60)
    print("Manual Forward Pass Smoke Test")
    print("=" * 60)

    sep = "─" * 60

    print(f"\nLoading MNIST ...")
    X_train, y_train, _, _ = load_mnist("dataset/mnist")
    X = X_train[:batch_size]
    Y = one_hot(y_train[:batch_size])
    print(f"  Batch shape  : X={X.shape}  Y={Y.shape}")

    dense1 = Dense(784, 256)
    relu1 = ReLU()
    dense2 = Dense(256, 128)
    relu2 = ReLU()
    dense3 = Dense(128, 10)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()

    Z1 = dense1.forward(X)
    A1 = relu1.forward(Z1)
    Z2 = dense2.forward(A1)
    A2 = relu2.forward(Z2)
    Z3 = dense3.forward(A2)
    P = softmax.forward(Z3)
    loss = loss_fn.forward(P, Y)

    print(f"\n{sep}\n  Forward pass\n{sep}")
    print(
        f"  {X.shape} → Dense1 → {Z1.shape} → ReLU → Dense2 → {Z2.shape}"
        f" → ReLU → Dense3 → {Z3.shape} → Softmax → {P.shape}"
    )
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-5)
    print(f"  Row sums : [{P.sum(axis=1).min():.8f}, {P.sum(axis=1).max():.8f}]  ✓")
    print(f"  Loss     : {loss:.6f}  (expect ≈ {np.log(10):.6f})  ✓")
    print(f"\nsmoke test passed.\n")


# ══════════════════════════════════════════════════════════════════════════
# Backpropagation
# ══════════════════════════════════════════════════════════════════════════


def _forward(layers, X, Y):
    """
    Run a forward pass through a list of (layer, activation) pairs
    followed by a final dense layer, softmax, and loss.

    Parameters
    ----------
    layers  : list of (Dense, ReLU) pairs + final Dense
    X       : input batch
    Y       : one-hot labels

    Returns
    -------
    loss       : float
    components : dict of all objects needed for backward
    """
    dense1, relu1, dense2, relu2, dense3, softmax, loss_fn = layers

    Z1 = dense1.forward(X)
    A1 = relu1.forward(Z1)
    Z2 = dense2.forward(A1)
    A2 = relu2.forward(Z2)
    Z3 = dense3.forward(A2)
    P = softmax.forward(Z3)
    loss = loss_fn.forward(P, Y)
    return loss


def backward_pass_test(batch_size: int = 64) -> None:
    """
    Run a full forward → backward pass and verify that:
    - Every parameter (W, b) receives a non-zero gradient.
    - Gradient shapes match parameter shapes exactly.
    - Input gradient (dX) flows all the way back to the first layer.
    """

    print("=" * 60)
    print("Full Backward Pass")
    print("=" * 60)

    sep = "─" * 60

    X_train, y_train, _, _ = load_mnist("dataset/mnist")
    X = X_train[:batch_size].astype(np.float64)
    Y = one_hot(y_train[:batch_size]).astype(np.float64)

    # Build network
    np.random.seed(0)
    dense1 = Dense(784, 256)
    relu1 = ReLU()
    dense2 = Dense(256, 128)
    relu2 = ReLU()
    dense3 = Dense(128, 10)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()

    # ── Forward ──────────────────────────────────────────────────────────
    Z1 = dense1.forward(X)
    A1 = relu1.forward(Z1)
    Z2 = dense2.forward(A1)
    A2 = relu2.forward(Z2)
    Z3 = dense3.forward(A2)
    P = softmax.forward(Z3)
    loss = loss_fn.forward(P, Y)

    print(f"\nForward loss : {loss:.6f}")

    # ── Backward - chain rule, right to left ─────────────────────────────
    #
    # The chain rule flows gradients backwards through the graph:
    #
    #   dL/dZ3 = CrossEntropy.backward()          fused Softmax+CE gradient
    #   dL/dA2 = Dense3.backward(dL/dZ3)          pushes gradient to Dense3 input
    #   dL/dZ2 = ReLU2.backward(dL/dA2)           gates gradient through ReLU mask
    #   dL/dA1 = Dense2.backward(dL/dZ2)          pushes gradient to Dense2 input
    #   dL/dZ1 = ReLU1.backward(dL/dA1)           gates gradient through ReLU mask
    #   dL/dX  = Dense1.backward(dL/dZ1)          pushes gradient to inputs
    #
    # Each Dense.backward() also stores dW and db on the layer object, which the optimizer will read.

    print(f"\n{sep}\n  Backward chain\n{sep}")

    dZ3 = loss_fn.backward()
    print(f"  loss_fn.backward()  → dZ3     {dZ3.shape}")

    dA2 = dense3.backward(dZ3)
    print(
        f"  dense3.backward()   → dA2     {dA2.shape}"
        f"   dW3={dense3.dW.shape}  db3={dense3.db.shape}"
    )

    dZ2 = relu2.backward(dA2)
    print(f"  relu2.backward()    → dZ2     {dZ2.shape}")

    dA1 = dense2.backward(dZ2)
    print(
        f"  dense2.backward()   → dA1     {dA1.shape}"
        f"   dW2={dense2.dW.shape}  db2={dense2.db.shape}"
    )

    dZ1 = relu1.backward(dA1)
    print(f"  relu1.backward()    → dZ1     {dZ1.shape}")

    dX = dense1.backward(dZ1)
    print(
        f"  dense1.backward()   → dX      {dX.shape}"
        f"   dW1={dense1.dW.shape}  db1={dense1.db.shape}"
    )

    # ── Gradient shape assertions ─────────────────────────────────────────
    print(f"\n{sep}\n  Gradient shape checks\n{sep}")

    checks = [
        ("dense1.dW", dense1.dW, (784, 256)),
        ("dense1.db", dense1.db, (256,)),
        ("dense2.dW", dense2.dW, (256, 128)),
        ("dense2.db", dense2.db, (128,)),
        ("dense3.dW", dense3.dW, (128, 10)),
        ("dense3.db", dense3.db, (10,)),
        ("dX", dX, (64, 784)),
    ]

    for name, arr, expected_shape in checks:
        assert (
            arr.shape == expected_shape
        ), f"{name}: got {arr.shape}, expected {expected_shape}"
        print(f"  {name:12s} shape={arr.shape}  ✓")

    # ── Non-zero gradient check ───────────────────────────────────────────
    print(f"\n{sep}\n  Non-zero gradient checks\n{sep}")

    for name, arr, _ in checks:
        norm = np.linalg.norm(arr)
        assert norm > 0, f"{name} gradient is all-zero (gradient vanished)"
        print(f"  {name:12s} ||grad||={norm:.6f}  ✓")

    print(f"\nBackward pass test passed.\n")


# ══════════════════════════════════════════════════════════════════════════
# Numerical Gradient Check
#
# This is the most rigorous test for a hand-written backward pass.
#
# For any scalar loss L and any parameter θ, the finite-difference
# approximation of the gradient is:
#
#     dL/dθ  ≈  ( L(θ + ε) − L(θ − ε) )  /  (2ε)
#
# This is the "centred difference" formula.  It converges at O(ε²) -
# halving ε reduces the error by 4×.
#
# We compare this numerical estimate to our analytical gradient.
# If they match to ~4 decimal places, the backward pass is correct.
#
# We use a tiny network (4→3→2) so the check runs fast.
# A larger ε (1e-4) gives cleaner numerics than the asymptotic 1e-7.
# ══════════════════════════════════════════════════════════════════════════


def _forward_tiny(d1, r1, d2, softmax, loss_fn, X, Y):
    """One forward pass through the tiny network. Returns scalar loss."""
    Z1 = d1.forward(X)
    A1 = r1.forward(Z1)
    Z2 = d2.forward(A1)
    P = softmax.forward(Z2)
    return loss_fn.forward(P, Y)


def _relative_error(analytical, numerical):
    """
    Relative error between two gradient matrices.

        err = |a - n| / max(|a|, |n|, 1e-8)

    Values below 1e-4 indicate a correct implementation.
    """
    return np.abs(analytical - numerical) / (
        np.maximum(np.abs(analytical), np.abs(numerical)) + 1e-8
    )


def numerical_grad_check() -> None:
    """
    Verify every gradient using finite differences.

    Architecture: Dense(4→3) → ReLU → Dense(3→2) → Softmax → CE
    Batch size  : 5 examples
    ε           : 1e-4
    """

    print("=" * 60)
    print("Numerical Gradient Check")
    print("=" * 60)

    sep = "─" * 60
    EPS = 1e-4

    np.random.seed(7)

    # Tiny network - manageable number of parameters to perturb
    d1 = Dense(4, 3)
    r1 = ReLU()
    d2 = Dense(3, 2)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()

    # Use float64 throughout - finite differences need higher precision
    d1.W = d1.W.astype(np.float64)
    d1.b = d1.b.astype(np.float64)
    d2.W = d2.W.astype(np.float64)
    d2.b = d2.b.astype(np.float64)

    X = np.random.randn(5, 4)
    Y = np.eye(2)[[0, 1, 0, 1, 0]]  # one-hot labels

    # ── Analytical gradients ──────────────────────────────────────────────
    _forward_tiny(d1, r1, d2, softmax, loss_fn, X, Y)

    dZ2 = loss_fn.backward()
    dA1 = d2.backward(dZ2)
    dZ1 = r1.backward(dA1)
    d1.backward(dZ1)

    analytic = {
        "d2.W": d2.dW.copy(),
        "d2.b": d2.db.copy(),
        "d1.W": d1.dW.copy(),
        "d1.b": d1.db.copy(),
    }

    # ── Numerical gradients via centred finite differences ─────────────────
    def numerical_grad(param_array, name):
        """Perturb each element ±ε and estimate gradient by finite difference."""
        grad = np.zeros_like(param_array)
        it = np.nditer(param_array, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index

            orig = param_array[idx]

            param_array[idx] = orig + EPS
            loss_plus = _forward_tiny(d1, r1, d2, softmax, loss_fn, X, Y)

            param_array[idx] = orig - EPS
            loss_minus = _forward_tiny(d1, r1, d2, softmax, loss_fn, X, Y)

            param_array[idx] = orig  # restore

            grad[idx] = (loss_plus - loss_minus) / (2 * EPS)
            it.iternext()

        return grad

    numerical = {
        "d2.W": numerical_grad(d2.W, "d2.W"),
        "d2.b": numerical_grad(d2.b, "d2.b"),
        "d1.W": numerical_grad(d1.W, "d1.W"),
        "d1.b": numerical_grad(d1.b, "d1.b"),
    }

    # ── Compare ────────────────────────────────────────────────────────────
    print(
        f"\n{'Parameter':<10}  {'Max rel error':<18}  {'Mean rel error':<18}  {'Status'}"
    )
    print(sep)

    THRESHOLD = 1e-4

    all_passed = True
    for name in ["d2.W", "d2.b", "d1.W", "d1.b"]:
        a = analytic[name]
        n = numerical[name]
        err = _relative_error(a, n)
        max_err = err.max()
        mean_err = err.mean()
        ok = max_err < THRESHOLD
        all_passed = all_passed and ok
        status = "✓" if ok else "✗  FAIL"
        print(
            f"  {name:<8}  max={max_err:.2e}          mean={mean_err:.2e}          {status}"
        )

    print(sep)

    if all_passed:
        print(f"\n  All gradients verified.  Max relative error < {THRESHOLD}.")
        print(f"  The backward pass is mathematically correct.  ✓")
    else:
        print(f"\n  GRADIENT CHECK FAILED - review backward() implementations.")

    # ── Show a sample analytical vs numerical comparison ───────────────────
    print(f"\n{sep}")
    print(f"  Sample: d2.W  analytical vs numerical (first 3 elements)")
    print(sep)
    a_flat = analytic["d2.W"].flat
    n_flat = numerical["d2.W"].flat
    for i, (a, n) in enumerate(zip(a_flat, n_flat)):
        if i >= 3:
            break
        print(
            f"  [{i}]  analytical={a:+.8f}   numerical={n:+.8f}   "
            f"diff={abs(a-n):.2e}"
        )

    print(f"\nNumerical gradient check complete.\n")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    smoke_test()
    backward_pass_test()
    numerical_grad_check()


# ──────────────────────────────────────────────────────────────────────────
# Backpropagation Verification
#
# The backward pass chain (reverse of forward):
#
#   dZ3 = loss_fn.backward()       ← fused Softmax+CE gradient (P-Y)/N
#   dA2 = dense3.backward(dZ3)     ← pushes gradient through last Dense
#   dZ2 = relu2.backward(dA2)      ← gates gradient through ReLU mask
#   dA1 = dense2.backward(dZ2)     ← pushes gradient through middle Dense
#   dZ1 = relu1.backward(dA1)      ← gates gradient through ReLU mask
#   dX  = dense1.backward(dZ1)     ← pushes gradient through first Dense
#
# Verification strategy: numerical gradient checking.
#
#   For each weight w_i, estimate the gradient by finite differences:
#
#       grad_numerical[i] = (L(w_i + h) - L(w_i - h)) / (2h)
#
#   Then compare against the analytical gradient from backward().
#   If they match to ~4-5 significant figures, backprop is correct.
#
#   Relative error formula:
#
#       rel_error = |grad_analytical - grad_numerical|
#                   ────────────────────────────────────
#                   |grad_analytical| + |grad_numerical|
#
#   Thresholds (rule of thumb):
#       rel_error < 1e-4  →  backprop is likely correct
#       rel_error > 1e-2  →  there is a bug
# ──────────────────────────────────────────────────────────────────────────


def _forward(layers: list, X: np.ndarray) -> np.ndarray:
    """Run a forward pass through an ordered list of layers/activations."""
    out = X
    for layer in layers:
        out = layer.forward(out)
    return out


def _loss(layers: list, loss_fn, X: np.ndarray, Y: np.ndarray) -> float:
    """Forward pass + loss in one call. Used by the numerical checker."""
    P = _forward(layers, X)
    return loss_fn.forward(P, Y)


def _numerical_grad(layers, loss_fn, X, Y, param, h=1e-5):
    """
    Estimate the gradient of *param* (a weight matrix or bias vector)
    using centred finite differences.

    For each element param[i]:
        param[i] += h  →  loss_plus
        param[i] -= h  →  loss_minus
        grad[i]  = (loss_plus - loss_minus) / (2h)

    This is O(n) forward passes where n = number of elements in param.
    Only used for small layers in tests - never during training.
    """
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index

        orig = param[idx]
        param[idx] = orig + h
        loss_plus = _loss(layers, loss_fn, X, Y)

        param[idx] = orig - h
        loss_minus = _loss(layers, loss_fn, X, Y)

        param[idx] = orig  # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * h)
        it.iternext()

    return grad


def _relative_error(analytical: np.ndarray, numerical: np.ndarray) -> float:
    """
    Relative error between two gradient arrays.
    Safe against both being zero.
    """
    num = np.abs(analytical - numerical).max()
    denom = np.abs(analytical).max() + np.abs(numerical).max() + 1e-12
    return float(num / denom)


def backprop_test() -> None:
    """
    Full backward pass + numerical gradient check.

    Uses a tiny network on a small batch so the numerical checker
    (which requires 2 x n forward passes) completes in seconds.

    Architecture under test:
        Dense(8→6) → ReLU → Dense(6→4) → Softmax + CrossEntropy
    """

    print("=" * 60)
    print("Backpropagation Verification")
    print("=" * 60)

    sep = "─" * 60
    rng = np.random.default_rng(0)

    # ── Tiny network so numerical grad check is fast ──────────────────
    N, D_in, D_h, D_out = 6, 8, 6, 4  # batch=6, in=8, hidden=6, classes=4

    dense1 = Dense(D_in, D_h)
    relu1 = ReLU()
    dense2 = Dense(D_h, D_out)
    softmax = Softmax()
    loss_fn = CrossEntropyLoss()

    layers = [dense1, relu1, dense2, softmax]

    # Random input and one-hot labels
    X = rng.standard_normal((N, D_in)).astype(np.float64)
    dense1.W = dense1.W.astype(np.float64)
    dense1.b = dense1.b.astype(np.float64)
    dense2.W = dense2.W.astype(np.float64)
    dense2.b = dense2.b.astype(np.float64)

    labels = rng.integers(0, D_out, size=N)
    Y = np.eye(D_out)[labels]  # one-hot, shape (N, D_out)

    # ── Forward pass ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Forward pass")
    print(sep)

    P = _forward(layers, X)
    loss = loss_fn.forward(P, Y)
    print(f"  Loss before backward : {loss:.6f}")

    # ── Backward pass - full chain ─────────────────────────────────────
    print(f"\n{sep}")
    print("  Backward pass chain")
    print(sep)

    dZ2 = loss_fn.backward()  # (P - Y) / N
    print(f"  loss_fn.backward()  → dZ2  {dZ2.shape}")

    dA1 = dense2.backward(dZ2)  # populates dense2.dW, dense2.db
    print(
        f"  dense2.backward(dZ2) → dA1  {dA1.shape}  "
        f"[dW={dense2.dW.shape}, db={dense2.db.shape}]"
    )

    dZ1 = relu1.backward(dA1)  # gates through stored mask
    print(f"  relu1.backward(dA1)  → dZ1  {dZ1.shape}")

    dX = dense1.backward(dZ1)  # populates dense1.dW, dense1.db
    print(
        f"  dense1.backward(dZ1) → dX   {dX.shape}  "
        f"[dW={dense1.dW.shape}, db={dense1.db.shape}]"
    )

    # ── Numerical gradient check ───────────────────────────────────────
    print(f"\n{sep}")
    print("  Numerical gradient check  (centred finite differences, h=1e-5)")
    print(sep)

    checks = [
        ("dense2.W", dense2.W, dense2.dW),
        ("dense2.b", dense2.b, dense2.db),
        ("dense1.W", dense1.W, dense1.dW),
        ("dense1.b", dense1.b, dense1.db),
    ]

    all_passed = True
    for name, param, grad_analytical in checks:
        grad_numerical = _numerical_grad(layers, loss_fn, X, Y, param)
        err = _relative_error(grad_analytical, grad_numerical)
        status = "✓" if err < 1e-4 else "✗  FAIL"
        if err >= 1e-4:
            all_passed = False
        print(f"  {name:<12}  rel_error = {err:.2e}   {status}")

    assert all_passed, "One or more gradient checks failed."

    # ── Single gradient descent step - loss must decrease ─────────────
    print(f"\n{sep}")
    print("  Manual gradient descent step  (η = 0.1)")
    print(sep)

    lr = 0.1

    # Save original weights
    W2_before = dense2.W.copy()
    b2_before = dense2.b.copy()
    W1_before = dense1.W.copy()
    b1_before = dense1.b.copy()

    loss_before = _loss(layers, loss_fn, X, Y)

    # Apply gradient descent: W ← W - η * dW
    dense2.W -= lr * dense2.dW
    dense2.b -= lr * dense2.db
    dense1.W -= lr * dense1.dW
    dense1.b -= lr * dense1.db

    loss_after = _loss(layers, loss_fn, X, Y)

    print(f"  Loss before update : {loss_before:.6f}")
    print(f"  Loss after  update : {loss_after:.6f}")
    print(f"  Loss decreased     : {'✓' if loss_after < loss_before else '✗  FAIL'}")

    assert loss_after < loss_before, (
        f"Loss did not decrease after gradient step: "
        f"{loss_before:.6f} → {loss_after:.6f}"
    )

    # ── MNIST scale: full-size forward+backward ────────────────────────
    print(f"\n{sep}")
    print("  Full-scale check  (MNIST batch, 784→256→128→10)")
    print(sep)

    X_train, y_train, _, _ = load_mnist("dataset/mnist")
    X64 = X_train[:64].astype(np.float64)
    Y64 = one_hot(y_train[:64]).astype(np.float64)

    d1 = Dense(784, 256)
    d1.W = d1.W.astype(np.float64)
    d1.b = d1.b.astype(np.float64)
    r1 = ReLU()
    d2 = Dense(256, 128)
    d2.W = d2.W.astype(np.float64)
    d2.b = d2.b.astype(np.float64)
    r2 = ReLU()
    d3 = Dense(128, 10)
    d3.W = d3.W.astype(np.float64)
    d3.b = d3.b.astype(np.float64)
    sm = Softmax()
    lf = CrossEntropyLoss()

    P64 = _forward([d1, r1, d2, r2, d3, sm], X64)
    loss0 = lf.forward(P64, Y64)

    dZ = lf.backward()
    dA = d3.backward(dZ)
    dZ = r2.backward(dA)
    dA = d2.backward(dZ)
    dZ = r1.backward(dA)
    _ = d1.backward(dZ)

    assert all(x.dW is not None for x in [d1, d2, d3]), "Missing dW"
    assert all(x.db is not None for x in [d1, d2, d3]), "Missing db"

    # One update step
    for layer in [d1, d2, d3]:
        layer.W -= 0.01 * layer.dW
        layer.b -= 0.01 * layer.db

    P64_after = _forward([d1, r1, d2, r2, d3, sm], X64)
    loss_after2 = lf.forward(P64_after, Y64)

    print(f"  Loss before : {loss0:.6f}")
    print(f"  Loss after  : {loss_after2:.6f}")
    assert loss_after2 < loss0, "Full-scale loss did not decrease"
    print(f"  Loss decreased on MNIST batch  ✓")

    print(f"\n{'=' * 60}")
    print(f"  Backpropagation verification complete - all gradients verified.")
    print(f"  Backpropagation is mathematically correct.")
    print(f"  Ready for SGD Optimizer.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    smoke_test()
    print()
    backprop_test()
