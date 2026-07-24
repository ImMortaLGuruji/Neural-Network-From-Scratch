"""
train.py
--------
Entry point for training the neural network on MNIST.

Usage
-----
    python train.py

What this script does
---------------------
1.  Loads the MNIST dataset (downloads if absent).
2.  Builds the 784 → 256 → 128 → 10 network.
3.  Trains for a configurable number of epochs using mini-batch SGD.
4.  Logs loss and accuracy to the console after every epoch.
5.  Saves the trained weights to saved_model/weights.npz.

Mini-batch training pipeline (one epoch)
-----------------------------------------
    shuffle training data
        │
        ▼
    for each mini-batch:
        zero_grad
            │
            ▼
        forward pass   →  probabilities P
            │
            ▼
        cross-entropy loss
            │
            ▼
        backward pass  →  gradients dW, db
            │
            ▼
        optimizer.step →  update W, b
        │
        ▼
    compute epoch loss & accuracy
        │
        ▼
    (repeat for next epoch)

Saved model
-----------
Weights are saved as a NumPy .npz archive after training.
evaluate.py and predict.py load from this file.
"""

import os
import time
import numpy as np

from nn.layers import Dense
from nn.activations import ReLU, Softmax
from nn.losses import CrossEntropyLoss
from nn.optimizer import SGD
from nn.network import Network
from nn.utils import load_mnist, one_hot

# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════

LEARNING_RATE = 0.1
EPOCHS = 15
BATCH_SIZE = 64
MOMENTUM = 0.0  # set to 0.9 to enable momentum SGD
HIDDEN_1 = 256
HIDDEN_2 = 128
DATA_DIR = "dataset/mnist"
SAVE_DIR = "saved_model"
WEIGHTS_FILE = os.path.join(SAVE_DIR, "weights.npz")


# ══════════════════════════════════════════════════════════════════════════
# Mini-batch generator
# ══════════════════════════════════════════════════════════════════════════


def iter_batches(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    Yield (X_batch, Y_batch) mini-batches from a dataset.

    Parameters
    ----------
    X          : np.ndarray, shape (N, D) - features
    Y          : np.ndarray, shape (N, C) - one-hot labels
    batch_size : int                      - examples per batch
    shuffle    : bool                     - shuffle before each epoch

    Yields
    ------
    X_batch : shape (batch_size, D)
    Y_batch : shape (batch_size, C)

    The last batch may be smaller than batch_size if N % batch_size != 0.
    """
    N = X.shape[0]
    indices = np.arange(N)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], Y[idx]


# ══════════════════════════════════════════════════════════════════════════
# Model builder
# ══════════════════════════════════════════════════════════════════════════


def build_network() -> Network:
    """
    Construct and return the feedforward network.

    Architecture
    ------------
        Input  (784)
          │
        Dense  784 → HIDDEN_1    He init
          │
        ReLU
          │
        Dense  HIDDEN_1 → HIDDEN_2   He init
          │
        ReLU
          │
        Dense  HIDDEN_2 → 10     He init
          │
        Softmax
          │
        CrossEntropyLoss  (during training only)
    """
    layers = [
        Dense(784, HIDDEN_1),
        ReLU(),
        Dense(HIDDEN_1, HIDDEN_2),
        ReLU(),
        Dense(HIDDEN_2, 10),
        Softmax(),
    ]
    loss_fn = CrossEntropyLoss()
    optimizer = SGD(layers, lr=LEARNING_RATE, momentum=MOMENTUM)
    return Network(layers, loss_fn, optimizer)


# ══════════════════════════════════════════════════════════════════════════
# Weight persistence
# ══════════════════════════════════════════════════════════════════════════


def save_weights(net: Network, path: str) -> None:
    """
    Save the weight and bias arrays of every Dense layer to a .npz file.

    Format
    ------
    The archive contains keys:
        layer_0_W, layer_0_b
        layer_1_W, layer_1_b
        ...
    where indices count Dense layers only.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {}
    dense_idx = 0
    for layer in net.layers:
        if isinstance(layer, Dense):
            arrays[f"layer_{dense_idx}_W"] = layer.W
            arrays[f"layer_{dense_idx}_b"] = layer.b
            dense_idx += 1
    np.savez(path, **arrays)


def load_weights(net: Network, path: str) -> None:
    """
    Restore weights from a .npz file into the Dense layers of *net*.

    The layer count and shapes must match the saved checkpoint exactly.
    """
    data = np.load(path)
    dense_idx = 0
    for layer in net.layers:
        if isinstance(layer, Dense):
            layer.W = data[f"layer_{dense_idx}_W"]
            layer.b = data[f"layer_{dense_idx}_b"]
            dense_idx += 1


# ══════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════


def train() -> None:
    """
    Full training run on MNIST.

    Prints one summary line per epoch:
        Epoch  1/15  loss=2.2874  train_acc=32.54%  val_acc=33.01%  (4.2s)

    Saves weights to WEIGHTS_FILE after the final epoch.
    """

    # ── Separator helpers ─────────────────────────────────────────────
    W = 72

    def header(title):
        print(f"\n{'═' * W}\n  {title}\n{'═' * W}")

    def sep():
        print("─" * W)

    # ── Dataset ───────────────────────────────────────────────────────
    header("Loading MNIST")
    X_train, y_train, X_val, y_val = load_mnist(DATA_DIR)
    Y_train = one_hot(y_train)

    print(f"  Train : {X_train.shape[0]:,} images  {X_train.shape[1]} features")
    print(f"  Val   : {X_val.shape[0]:,} images")
    print(f"  Classes : 0-9  ({Y_train.shape[1]} outputs)")

    # ── Network ───────────────────────────────────────────────────────
    header("Network")
    net = build_network()
    net.summary()

    # ── Hyperparameter summary ────────────────────────────────────────
    header("Hyperparameters")
    print(f"  epochs        : {EPOCHS}")
    print(f"  batch_size    : {BATCH_SIZE}")
    print(f"  learning_rate : {LEARNING_RATE}")
    print(f"  momentum      : {MOMENTUM}")
    n_batches = int(np.ceil(len(X_train) / BATCH_SIZE))
    print(f"  batches/epoch : {n_batches}")

    # ── Training ──────────────────────────────────────────────────────
    header("Training")

    col = f"  {'Epoch':>5}  {'Loss':>8}  {'Train acc':>10}  {'Val acc':>8}  {'Time':>6}"
    print(col)
    sep()

    history = {"loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    np.random.seed(11)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        epoch_losses = []

        # ── one epoch: iterate all mini-batches ───────────────────────
        for X_batch, Y_batch in iter_batches(X_train, Y_train, BATCH_SIZE):
            loss = net.train_step(X_batch, Y_batch)
            epoch_losses.append(loss)

        # ── epoch metrics ─────────────────────────────────────────────
        mean_loss = float(np.mean(epoch_losses))

        # Accuracy on the full training set — evaluated in one pass,
        # batched so we don't blow memory on 60k × 784
        train_preds = np.concatenate(
            [net.predict(X_train[i : i + 512]) for i in range(0, len(X_train), 512)]
        )
        train_acc = float((train_preds == y_train).mean())

        val_preds = np.concatenate(
            [net.predict(X_val[i : i + 512]) for i in range(0, len(X_val), 512)]
        )
        val_acc = float((val_preds == y_val).mean())

        elapsed = time.time() - t0

        history["loss"].append(mean_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        marker = "  ◀ best" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"  {epoch:>5}/{EPOCHS}"
            f"  {mean_loss:>8.4f}"
            f"  {train_acc:>10.2%}"
            f"  {val_acc:>8.2%}"
            f"  {elapsed:>5.1f}s"
            f"{marker}"
        )

    # ── Final summary ─────────────────────────────────────────────────
    sep()
    print(f"\n  Best val accuracy : {best_val_acc:.2%}")
    print(f"  Final train loss  : {history['loss'][-1]:.4f}")

    # ── Loss curve (ASCII) ────────────────────────────────────────────
    header("Loss curve")
    losses = history["loss"]
    lo, hi = min(losses), max(losses)
    bar_w = 40
    for i, l in enumerate(losses):
        filled = int((hi - l) / (hi - lo + 1e-9) * bar_w)
        bar = "█" * filled + "░" * (bar_w - filled)
        print(f"  Epoch {i+1:>2}  [{bar}]  {l:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    header("Saving weights")
    save_weights(net, WEIGHTS_FILE)
    print(f"  Weights saved → {WEIGHTS_FILE}")
    print(f"  Keys: {list(np.load(WEIGHTS_FILE).keys())}")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train()
