"""
network.py
----------
Network class - assembles layers, loss, and optimizer into a trainable model.

Usage
-----
    from nn.network import Network
    from nn.layers import Dense
    from nn.activations import ReLU, Softmax
    from nn.losses import CrossEntropyLoss
    from nn.optimizer import SGD

    layers = [Dense(784, 256), ReLU(), Dense(256, 128), ReLU(), Dense(128, 10), Softmax()]
    net = Network(layers, CrossEntropyLoss(), SGD(layers, lr=0.1))

    loss  = net.train_step(X_batch, Y_batch)
    preds = net.predict(X_test)
    acc   = net.accuracy(X_test, y_test)
    net.summary()
"""

import numpy as np
from nn.layers import Dense
from nn.activations import Softmax
from nn.losses import CrossEntropyLoss
from nn.optimizer import SGD


class Network:
    """
    Feedforward neural network.

    Wraps an ordered list of layers and activations, a loss function, and
    an optimizer into a single object with a clean training API.

    Design note: Softmax and the backward pass
    -------------------------------------------
    Softmax must be the final entry in `layers`. It is included in the
    forward pass but skipped during backward.

    Why? CrossEntropyLoss.backward() returns the fused Softmax + Cross
    Entropy gradient (P - Y) / N, which is already the gradient with
    respect to the logits entering Softmax - not with respect to its
    output. So the backward chain starts at the Dense layer before
    Softmax, not at Softmax itself.

    Calling Softmax.backward() on top of the fused gradient would
    apply the transformation twice and produce wrong gradients.

    Parameters
    ----------
    layers    : list
        Ordered layer and activation objects. Must end with Softmax.
    loss_fn   : CrossEntropyLoss
        Loss function paired with the Softmax output layer.
    optimizer : SGD
        Optimizer already initialised with the Dense layers and
        learning rate.
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

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Run a forward pass through every layer.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)

        Returns
        -------
        P : np.ndarray, shape (N, num_classes)
            Softmax probability distribution over classes.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _backward(self, dZ: np.ndarray) -> None:
        """
        Run the backward chain through all layers except the final Softmax.

        Parameters
        ----------
        dZ : np.ndarray, shape (N, num_classes)
            Fused Softmax + Cross Entropy gradient from loss_fn.backward().
            This is already the gradient w.r.t. the pre-Softmax logits,
            so Softmax (layers[-1]) is excluded from the chain.
        """
        grad = dZ
        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad)

    def train_step(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        One complete training iteration.

        Runs: zero_grad → forward → loss → backward → optimizer step.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)   input batch
        Y : np.ndarray, shape (N, C)       one-hot labels

        Returns
        -------
        loss : float
            Mean cross-entropy loss over the batch.
        """
        self.optimizer.zero_grad()

        P = self.forward(X)
        loss = self.loss_fn.forward(P, Y)

        dZ = self.loss_fn.backward()
        self._backward(dZ)

        self.optimizer.step()

        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return the predicted class index for each example.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)

        Returns
        -------
        np.ndarray of shape (N,) - argmax of the Softmax output.
        """
        return self.forward(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.

        Parameters
        ----------
        X : np.ndarray, shape (N, D_in)   input features
        y : np.ndarray, shape (N,)         integer class labels

        Returns
        -------
        float in [0, 1] - fraction of correctly classified examples.
        """
        return float((self.predict(X) == y).mean())

    def summary(self) -> None:
        """Print the network architecture and total parameter count."""
        print("Network")
        print("─" * 40)
        total = 0
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                n = layer.W.size + layer.b.size
                total += n
                print(f"  [{i}] {layer}   params={n:,}")
            else:
                print(f"  [{i}] {layer}")
        print("─" * 40)
        print(f"  Total trainable parameters: {total:,}")

    def __repr__(self) -> str:
        names = " → ".join(type(l).__name__ for l in self.layers)
        return f"Network([{names}])"
