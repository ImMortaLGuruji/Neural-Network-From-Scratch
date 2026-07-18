"""
losses.py
---------
Cross Entropy loss for multi-class classification.

What is a loss function?
------------------------
After the network makes a prediction, we need a single number that
answers "how wrong was that prediction?"  That number is the loss.

Training is simply the process of making the loss as small as possible.

Cross Entropy Loss
------------------
Cross entropy measures the difference between two probability
distributions: the predicted distribution P and the true distribution Y.

    L = - (1/N) * sum_n  sum_c  Y[n,c] * log(P[n,c])

Because Y is one-hot (exactly one 1 per row, rest are 0), this reduces
to picking out just the log-probability of the correct class:

    L = - (1/N) * sum_n  log(P[n, true_class_n])

Intuition
---------
If the network is very confident and correct (P → 1 for true class):
    log(1) = 0  →  loss is small

If the network is very confident but wrong (P → 0 for true class):
    log(0) = -inf  →  loss is huge

Cross entropy penalises confident wrong answers far more harshly
than uncertain wrong answers.  This is exactly the behaviour we want.

Why not Mean Squared Error?
----------------------------
MSE works for regression but behaves poorly for classification:
- The gradient becomes very small when the network is confidently wrong
  (the "vanishing gradient" problem for outputs).
- Cross entropy avoids this because log has an infinite slope at 0.

Fused Softmax + Cross Entropy backward
---------------------------------------
The combined backward gradient is unusually simple:

    dL/dZ = (P - Y) / N

where Z is the pre-softmax logit matrix.  This fused result avoids
computing the full Softmax Jacobian.  Derivation is commented in backward().
"""

import numpy as np


class CrossEntropyLoss:
    """
    Categorical Cross Entropy loss, assuming Softmax output.

    Usage
    -----
    >>> loss_fn = CrossEntropyLoss()
    >>> loss = loss_fn.forward(P, Y)      # P: probabilities, Y: one-hot
    >>> dZ   = loss_fn.backward()         # returns dL/dZ (pre-softmax gradient)

    The backward pass returns the gradient w.r.t. the *logits* Z (before
    Softmax), not w.r.t. P.  This fused Softmax+CE gradient is what gets
    passed into the last Dense layer's backward().
    """

    def __init__(self) -> None:
        # Cached during forward() for use in backward()
        self._P: np.ndarray | None = None  # softmax probabilities
        self._Y: np.ndarray | None = None  # one-hot labels
        self._N: int | None = None  # batch size

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, P: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute the mean Cross Entropy loss over a batch.

        Parameters
        ----------
        P : np.ndarray, shape (N, C)
            Softmax output - probability distribution over C classes
            for each of the N examples.
        Y : np.ndarray, shape (N, C)
            One-hot encoded ground truth labels.
            Exactly one 1 per row, rest 0.

        Returns
        -------
        loss : float
            Mean cross entropy loss across the batch.

        Implementation detail - clipping
        ---------------------------------
        log(0) is undefined (-inf).  Even though Softmax never outputs
        exactly 0, floating-point underflow can produce 0.0 for very
        negative logits.

        Clipping P to [epsilon, 1] prevents log(0) without meaningfully
        affecting the loss for any realistic probability.
        """
        assert P.shape == Y.shape, (
            f"CrossEntropyLoss: P and Y shape mismatch - " f"P={P.shape}, Y={Y.shape}"
        )

        N = P.shape[0]

        # Cache for backward()
        self._P = P
        self._Y = Y
        self._N = N

        # Clip to avoid log(0)
        eps = 1e-12
        P_clipped = np.clip(P, eps, 1.0)

        # Full formula:  L = -(1/N) * sum( Y * log(P) )
        #
        # Since Y is one-hot, Y * log(P) is non-zero only at the
        # true class column.  The sum across classes collapses to
        # a single log-probability per example.
        loss = -np.sum(Y * np.log(P_clipped)) / N
        return float(loss)

    # ------------------------------------------------------------------
    # Backward pass  (fused Softmax + Cross Entropy)
    # ------------------------------------------------------------------

    def backward(self) -> np.ndarray:
        """
        Return the gradient of the loss w.r.t. the pre-softmax logits Z.

        Derivation
        ----------
        Let Z be the logit matrix, P = Softmax(Z), and
        L = CrossEntropy(P, Y).

        Unpack the chain rule:

            dL/dZ = dL/dP * dP/dZ

        dL/dP
        ~~~~~
        From the loss formula L = -(1/N) sum_n sum_c Y[n,c] log(P[n,c]):

            dL/dP[n,c] = -(1/N) * Y[n,c] / P[n,c]

        dP/dZ  (Softmax Jacobian)
        ~~~~~~~~~~~~~~~~~~~~~~~~~
        For the i-th output and j-th input of a Softmax row:

            dP_i/dZ_j = P_i * (delta_ij - P_j)

        where delta_ij = 1 if i == j, else 0.

        Combining (after summing over the Jacobian rows):

            dL/dZ[n,c] = sum_i  dL/dP[n,i] * dP[n,i]/dZ[n,c]

                       = sum_i  [-(1/N) * Y[n,i]/P[n,i]]
                                 * P[n,i] * (delta_ic - P[n,c])

                       = -(1/N) * sum_i  Y[n,i] * (delta_ic - P[n,c])

                       = -(1/N) * [Y[n,c] - P[n,c] * sum_i Y[n,i]]

        Since Y is one-hot, sum_i Y[n,i] = 1, giving:

            dL/dZ[n,c] = -(1/N) * (Y[n,c] - P[n,c])
                       =  (1/N) * (P[n,c] - Y[n,c])

        Or in matrix form:

            dL/dZ = (P - Y) / N

        This is remarkably clean.  The entire Softmax Jacobian collapsed
        away, leaving just the difference between predictions and labels.

        Returns
        -------
        dZ : np.ndarray, shape (N, C)
             Gradient of the loss w.r.t. the logits Z.
             This is passed directly into Dense.backward() for the last layer.
        """
        assert (
            self._P is not None
        ), "CrossEntropyLoss.backward() called before forward()."

        dZ = (self._P - self._Y) / self._N
        return dZ

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"
