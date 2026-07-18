"""
activations.py
--------------
Activation functions: ReLU and Softmax.

Why activation functions exist
-------------------------------
A sequence of purely linear layers collapses into a single linear
transformation, no matter how many layers are stacked:

    A(B(Cx)) == (ABC)x

Activation functions break this by introducing non-linearity between
layers.  Without them, the entire network is equivalent to logistic
regression - no matter its depth.

Modules
-------
ReLU    - element-wise, used after every hidden Dense layer.
Softmax - applied to the final layer output to produce probabilities.

Backward pass
-------------
Each class caches what it needs during forward()
so backward() has everything it requires.
"""

import numpy as np

# ============================================================
# ReLU
# ============================================================


class ReLU:
    """
    Rectified Linear Unit activation.

    Forward
    -------
        ReLU(x) = max(0, x)

    Applied element-wise.  Every negative value is replaced with zero;
    positive values pass through unchanged.

        Input : [-3,  2, -1, 5]
        Output: [ 0,  2,  0, 5]

    Why ReLU?
    ---------
    * Simple and cheap to compute - just a threshold.
    * Does not saturate for positive inputs, so gradients don't vanish
      in the positive region the way they do with sigmoid/tanh.
    * Empirically works well in practice for deep networks.

    Backward
    -------------------------
        dL/dX = dL/dA * (X > 0)

    The gradient of ReLU is 1 where the input was positive, 0 elsewhere.
    This binary mask is called the "ReLU gate".  The cache stores it
    during forward() so backward() can apply it without re-computing.
    """

    def __init__(self) -> None:
        # Stores the binary activation mask: True where X > 0
        # Shape matches the input; populated during forward().
        self._mask: np.ndarray | None = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply ReLU element-wise.

        Parameters
        ----------
        X : np.ndarray, any shape
            Pre-activation values from a Dense layer.

        Returns
        -------
        A : np.ndarray, same shape as X
            Post-activation values; all negative entries zeroed out.
        """
        # Cache the mask for backward()
        # True  where X > 0  →  gradient passes through
        # False where X ≤ 0  →  gradient is blocked (dead neuron)
        self._mask = X > 0

        # np.maximum broadcasts over any shape without allocation overhead
        return np.maximum(0, X)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backpropagate through ReLU.

        The derivative of ReLU(x) is:
            1  if x > 0
            0  if x ≤ 0

        So the upstream gradient dL/dA is passed through unchanged
        where the input was positive, and blocked (zeroed) where it
        was non-positive.

        Parameters
        ----------
        dA : np.ndarray, same shape as the forward input
             Gradient of the loss w.r.t. the ReLU output.

        Returns
        -------
        dX : np.ndarray, same shape
             Gradient of the loss w.r.t. the ReLU input.
        """
        assert self._mask is not None, "ReLU.backward() called before forward()."
        # Multiply upstream gradient by the gate
        return dA * self._mask

    def __repr__(self) -> str:
        return "ReLU()"


# ============================================================
# Softmax
# ============================================================


class Softmax:
    """
    Softmax activation - converts raw logits into a probability distribution.

    Forward
    -------
        P_i = exp(z_i) / sum_j( exp(z_j) )

    Properties
    ----------
    * Every output is in (0, 1).
    * All outputs sum to exactly 1.
    * The highest logit always gets the highest probability.
    * Softmax amplifies differences - it is "winner-takes-most".

    Numerical stability
    -------------------
    Raw exp() overflows for large logits:

        exp(1000) → inf

    A standard fix: subtract the row-wise maximum before exponentiating.
    This is mathematically equivalent because the constant cancels:

        exp(z_i - C) / sum_j( exp(z_j - C) )
        = exp(z_i) * exp(-C) / ( sum_j(exp(z_j)) * exp(-C) )
        = exp(z_i) / sum_j( exp(z_j) )               ← same result

    With C = max(z), every exponent is ≤ 0, so no overflow.

    Backward
    -------------------------
    In practice the Softmax backward is never computed in isolation.
    It is always fused with Cross Entropy loss, which simplifies the
    combined gradient to:

        dL/dZ = P - Y

    That derivation lives in losses.py, not here.
    The Softmax class caches its output P so the loss can use it.
    """

    def __init__(self) -> None:
        # Stores the output probabilities from the last forward pass.
        # Shape: (batch_size, num_classes).
        self._P: np.ndarray | None = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply Softmax row-wise.

        Parameters
        ----------
        Z : np.ndarray, shape (batch_size, num_classes)
            Raw logits from the final Dense layer.

        Returns
        -------
        P : np.ndarray, shape (batch_size, num_classes)
            Probability distribution over classes; each row sums to 1.
        """
        assert (
            Z.ndim == 2
        ), f"Softmax expects 2-D input (batch, classes), got shape {Z.shape}"

        # Subtract row-wise max for numerical stability (see docstring)
        # keepdims=True allows broadcasting back across the class axis
        shift = Z - Z.max(axis=1, keepdims=True)
        exp_z = np.exp(shift)

        # Normalise each row so it sums to 1
        P = exp_z / exp_z.sum(axis=1, keepdims=True)

        # Cache for backward() / loss computation
        self._P = P
        return P

    def __repr__(self) -> str:
        return "Softmax()"
