"""
layers.py
---------
Dense (fully connected) layer.

A Dense layer computes:

    Z = X @ W + b

where
    X - input matrix,  shape (batch_size, in_features)
    W - weight matrix, shape (in_features, out_features)
    b - bias vector,   shape (out_features,)
    Z - output matrix, shape (batch_size, out_features)

Weight Initialisation
---------------------
Weights are sampled from a scaled normal distribution:

    W ~ Normal(0, sqrt(2 / in_features))

This is He initialisation - the factor sqrt(2 / in) keeps the variance
of activations roughly constant through layers that use ReLU.

Without this scaling, activations either vanish or explode as the
signal passes through many layers.

Biases are initialised to zero.  This is standard: biases do not
suffer from symmetry breaking problems the way weights do.

Backward pass
-------------
Once the loss and activations are in place.
The layer caches the input X during forward() so backward() can use it.
"""

import numpy as np


class Dense:
    """
    A single fully-connected (linear) layer.

    Parameters
    ----------
    in_features  : int - number of input neurons (columns in X).
    out_features : int - number of output neurons (columns in Z).

    Attributes
    ----------
    W  : np.ndarray, shape (in_features, out_features)
         Learnable weight matrix.
    b  : np.ndarray, shape (out_features,)
         Learnable bias vector.
    _X : np.ndarray or None
         Input cached during the last forward pass.
         Used by backward().
    dW : np.ndarray or None
         Gradient of the loss w.r.t. W. Populated by backward().
    db : np.ndarray or None
         Gradient of the loss w.r.t. b. Populated by backward().
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features = in_features
        self.out_features = out_features

        # ------------------------------------------------------------------
        # He initialisation
        # ------------------------------------------------------------------
        # Draw weights from N(0, 1) then scale by sqrt(2 / in_features).
        #
        # Why sqrt(2 / in)?
        #   For a ReLU neuron roughly half of its inputs are zeroed out,
        #   so the effective fan-in is ~in/2.  Multiplying by 2 compensates.
        #
        # This keeps Var(Z) ≈ Var(X) regardless of layer width, which lets
        # gradients flow without vanishing or exploding.
        # ------------------------------------------------------------------
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale

        # Biases start at zero - no symmetry-breaking needed here
        self.b = np.zeros(out_features, dtype=np.float32)

        # Gradient placeholders - filled in Stage 6
        self._X: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Z = X @ W + b.

        Parameters
        ----------
        X : np.ndarray, shape (batch_size, in_features)
            Input activations from the previous layer (or the raw data).

        Returns
        -------
        Z : np.ndarray, shape (batch_size, out_features)
            Pre-activation output - also called "logits" at the final layer.

        Shape check
        -----------
        Asserts that X has the right number of columns so mistakes are caught
        immediately instead of silently producing garbage results.
        """
        assert X.ndim == 2, (
            f"Dense.forward expects a 2-D input (batch, features), "
            f"got shape {X.shape}"
        )
        assert X.shape[1] == self.in_features, (
            f"Dense.forward: X has {X.shape[1]} features "
            f"but layer expects {self.in_features}"
        )

        # Cache the input so backward() can compute dW = X.T @ dZ
        self._X = X

        # Core computation:
        #   X  (batch, in)  @  W  (in, out)  +  b  (out,)
        #   =  Z  (batch, out)
        #
        # NumPy broadcasts b across the batch dimension automatically.
        Z = X @ self.W + self.b
        return Z

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Backpropagate the gradient through this layer.

        Given the upstream gradient dL/dZ this method computes:

            dL/dW = X.T @ dZ          shape (in_features, out_features)
            dL/db = sum(dZ, axis=0)   shape (out_features,)
            dL/dX = dZ @ W.T          shape (batch_size, in_features)

        The derivations are:

            Z = X @ W + b

            dL/dW[i,j] = sum_n  dL/dZ[n,j]  *  dZ[n,j]/dW[i,j]
                       = sum_n  dL/dZ[n,j]  *  X[n,i]
                       = X.T @ dZ            (matrix form)

            dL/db[j]   = sum_n  dL/dZ[n,j]  *  1
                       = dZ.sum(axis=0)

            dL/dX[n,i] = sum_j  dL/dZ[n,j]  *  dZ[n,j]/dX[n,i]
                       = sum_j  dL/dZ[n,j]  *  W[i,j]
                       = dZ @ W.T           (matrix form)

        Parameters
        ----------
        dZ : np.ndarray, shape (batch_size, out_features)
             Gradient of the loss w.r.t. this layer's output Z.

        Returns
        -------
        dX : np.ndarray, shape (batch_size, in_features)
             Gradient of the loss w.r.t. this layer's input X.
             Passed to the previous layer during backprop.
        """
        assert (
            self._X is not None
        ), "backward() called before forward() - no cached input."

        self.dW = self._X.T @ dZ

        self.db = dZ.sum(axis=0)

        # Gradient w.r.t. input: pass upstream
        dX = dZ @ self.W.T
        return dX

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Dense(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"W={self.W.shape}, b={self.b.shape})"
        )
