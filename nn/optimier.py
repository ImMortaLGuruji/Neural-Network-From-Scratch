"""
optimizer.py
------------
Stochastic Gradient Descent (SGD) optimizer.

What is an optimizer?
----------------------
After backpropagation computes dW and db for every layer, something has to
actually apply those gradients to the weights. That is the optimizer's job.

Without an optimizer you'd write this manually every training step:

    layer.W -= lr * layer.dW
    layer.b -= lr * layer.db

The optimizer wraps that logic so the training loop stays clean:

    optimizer.step(layers)

SGD Update Rule
---------------
    W ← W - η * dW
    b ← b - η * db

where η (eta) is the learning rate - a small positive scalar that
controls how large each update is.

Why SGD?
--------
SGD is the foundation of nearly every optimizer in modern deep
learning.  Adam, RMSProp, and Momentum are all refinements of SGD.
Understanding vanilla SGD first makes those extensions clear.

Learning rate intuition
-----------------------
    η too small  →  updates are tiny, training crawls
    η too large  →  updates overshoot, loss oscillates or diverges
    η just right →  loss decreases steadily each step

Typical starting values for this network: 0.01 - 0.1.

Momentum
--------
Momentum accumulates a velocity vector in the direction of persistent
gradient, dampening oscillations and accelerating in consistent directions:

    v ← μ * v - η * dW
    W ← W + v

where μ is the momentum coefficient (typically 0.9).

Momentum is implemented here as an optional extension
set momentum=0.0 (default) for vanilla SGD.
"""

import numpy as np
from nn.layers import Dense


class SGD:
    """
    Stochastic Gradient Descent optimizer with optional momentum.

    Parameters
    ----------
    layers      : list[Dense]
        The Dense layers whose weights will be updated.
        (Activation layers have no parameters and are skipped.)
    lr          : float
        Learning rate η.  Controls the size of each weight update.
    momentum    : float, optional (default 0.0)
        Momentum coefficient μ.  Set to 0.0 for vanilla SGD.
        A value of 0.9 is typical when momentum is used.

    Usage
    -----
    >>> optimizer = SGD(layers=[dense1, dense2, dense3], lr=0.01)
    >>> # inside training loop:
    >>> optimizer.step()   # applies W ← W - lr * dW for every layer
    >>> optimizer.zero_grad()  # clears dW and db before next forward pass
    """

    def __init__(
        self,
        layers: list,
        lr: float = 0.01,
        momentum: float = 0.0,
    ) -> None:

        # Keep only layers that actually have learnable parameters
        self.layers = [l for l in layers if isinstance(l, Dense)]
        self.lr = lr
        self.momentum = momentum

        # Velocity buffers for momentum - one per (W, b) pair.
        # Initialised to zero; only allocated when momentum > 0.
        if momentum > 0.0:
            self._vW = [np.zeros_like(l.W) for l in self.layers]
            self._vb = [np.zeros_like(l.b) for l in self.layers]
        else:
            self._vW = None
            self._vb = None

    # ------------------------------------------------------------------
    # Step - apply gradients
    # ------------------------------------------------------------------

    def step(self) -> None:
        """
        Apply one gradient descent update to every Dense layer.

        Vanilla SGD (momentum == 0.0):
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            W ← W - lr * dW
            b ← b - lr * db

        SGD with momentum (momentum > 0.0):
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            vW ← μ * vW + lr * dW     (accumulate velocity)
            W  ← W - vW               (apply velocity)

            vb ← μ * vb + lr * db
            b  ← b - vb

        Raises
        ------
        AssertionError
            If a layer's gradients are None - i.e., backward() was
            never called before step().  This catches the common mistake
            of calling step() without first running a backward pass.
        """
        for i, layer in enumerate(self.layers):
            assert layer.dW is not None, (
                f"SGD.step(): layer {i} ({layer}) has dW=None. "
                "Did you call backward() before step()?"
            )
            assert layer.db is not None, (
                f"SGD.step(): layer {i} ({layer}) has db=None. "
                "Did you call backward() before step()?"
            )

            if self.momentum > 0.0:
                # Momentum update
                # v ← μ*v + lr*dW   (velocity accumulates gradient history)
                # W ← W - v         (step in the velocity direction)
                self._vW[i] = self.momentum * self._vW[i] + self.lr * layer.dW
                self._vb[i] = self.momentum * self._vb[i] + self.lr * layer.db
                layer.W -= self._vW[i]
                layer.b -= self._vb[i]
            else:
                # Vanilla SGD
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

    # ------------------------------------------------------------------
    # Zero gradients
    # ------------------------------------------------------------------

    def zero_grad(self) -> None:
        """
        Clear accumulated gradients in all layers.

        Call this at the start of every training step, before forward().

        Why is this necessary?
        ----------------------
        NumPy arrays are mutable.  If you do not clear dW and db between
        steps, the next call to backward() will compute new gradients and
        overwrite them - but if backward() is accidentally skipped, stale
        gradients from the previous step would be used silently.

        Zeroing explicitly makes the bug visible immediately (step()
        raises AssertionError if dW is None).

        The convention is:
            zero_grad → forward → loss → backward → step → repeat
        """
        for layer in self.layers:
            layer.dW = None
            layer.db = None

    # ------------------------------------------------------------------
    # Learning rate schedule
    # ------------------------------------------------------------------

    def set_lr(self, new_lr: float) -> None:
        """
        Update the learning rate mid-training.

        Useful for manual learning rate schedules, e.g.:
            - Decay by x0.1 after a fixed number of epochs.
            - Warm-up: start small and increase during early steps.

        Parameters
        ----------
        new_lr : float
            The new learning rate to use from the next step onward.
        """
        self.lr = new_lr

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.lr}, momentum={self.momentum}, "
            f"layers={len(self.layers)})"
        )
