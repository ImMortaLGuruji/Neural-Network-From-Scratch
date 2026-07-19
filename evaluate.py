"""
evaluate.py
-----------
Load trained weights and run a full evaluation on the MNIST test set.

Usage
-----
    python evaluate.py

Requires
--------
    saved_model/weights.npz - produced by train.py

Output
------
- Overall test accuracy
- Per-class accuracy table
- Confusion matrix (10 x 10)
- Most-confused class pairs
- Five hardest examples (highest-loss correct predictions)
- Five failure cases (wrong predictions with high confidence)
"""

import numpy as np
from nn.network import Network
from nn.utils import load_mnist, one_hot
from train import build_network, load_weights, WEIGHTS_FILE, DATA_DIR

DIGIT_NAMES = [str(d) for d in range(10)]
W = 72  # console width


def _sep(char="─"):
    print(char * W)


def _header(title):
    print(f"\n{'═' * W}\n  {title}\n{'═' * W}")


# ══════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════


def predict_batched(net: Network, X: np.ndarray, batch_size: int = 512):
    """Run forward pass in chunks to avoid memory pressure on large sets."""
    probs = np.concatenate(
        [net.forward(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]
    )
    return probs  # shape (N, 10)


# ══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════


def _ascii_digit(pixels: np.ndarray, width: int = 28) -> str:
    """Render a 784-vector as a 28x28 ASCII block."""
    ramp = " .:-=+*#%@"
    image = pixels.reshape(width, width)
    lines = []
    for row in image:
        lines.append("".join(ramp[int(v * (len(ramp) - 1))] for v in row))
    return "\n".join(lines)


def _bar(value: float, width: int = 30) -> str:
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def _confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = 10
) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ══════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════


def evaluate() -> None:

    # ── Load data ─────────────────────────────────────────────────────
    _header("Loading MNIST test set")
    _, _, X_test, y_test = load_mnist(DATA_DIR)
    print(f"  {len(X_test):,} test images x 784 features")

    # ── Load model ────────────────────────────────────────────────────
    _header("Loading trained weights")
    net = build_network()
    load_weights(net, WEIGHTS_FILE)
    net.summary()

    # ── Full-set inference ────────────────────────────────────────────
    _header("Running inference")
    P = predict_batched(net, X_test)  # (10000, 10)
    y_pred = P.argmax(axis=1)  # (10000,)
    Y_test = one_hot(y_test)
    losses = -(Y_test * np.log(np.clip(P, 1e-12, 1))).sum(axis=1)  # per-sample CE

    correct = y_pred == y_test
    acc = correct.mean()
    n_right = correct.sum()
    n_wrong = len(y_test) - n_right

    print(f"  Correct   : {n_right:,} / {len(y_test):,}")
    print(f"  Wrong     : {n_wrong:,}")
    print(f"  Accuracy  : {acc:.4%}")

    # ── Per-class accuracy ────────────────────────────────────────────
    _header("Per-class accuracy")
    print(f"  {'Digit':>5}  {'Correct':>7}  {'Total':>6}  {'Accuracy':>8}  Bar")
    _sep()
    for cls in range(10):
        mask = y_test == cls
        n_total = mask.sum()
        n_ok = (y_pred[mask] == cls).sum()
        cls_acc = n_ok / n_total
        bar = _bar(cls_acc, 28)
        print(f"    {cls:>3}    {n_ok:>6,}   {n_total:>5,}    {cls_acc:>7.2%}  {bar}")

    # ── Confusion matrix ──────────────────────────────────────────────
    _header("Confusion matrix  (rows = true, columns = predicted)")
    cm = _confusion_matrix(y_test, y_pred)

    header_row = "       " + "  ".join(f"{c:>4}" for c in range(10))
    print(header_row)
    _sep()
    for true_cls in range(10):
        row_parts = []
        for pred_cls in range(10):
            val = cm[true_cls, pred_cls]
            if true_cls == pred_cls:
                row_parts.append(f"\033[92m{val:>4}\033[0m")  # green on diagonal
            elif val > 0:
                row_parts.append(f"\033[91m{val:>4}\033[0m")  # red for errors
            else:
                row_parts.append(f"{'':>4}")
        print(f"  True {true_cls} │ " + "  ".join(row_parts))

    # ── Most-confused pairs ───────────────────────────────────────────
    _header("Most-confused class pairs  (off-diagonal, highest counts)")
    off = [
        (cm[t, p], t, p)
        for t in range(10)
        for p in range(10)
        if t != p and cm[t, p] > 0
    ]
    off.sort(reverse=True)
    print(f"  {'True':>5}  {'Predicted':>9}  {'Count':>6}  Interpretation")
    _sep()
    for count, true_cls, pred_cls in off[:8]:
        print(
            f"    {true_cls:>3}   →   {pred_cls:>3}          {count:>5}   "
            f"'{true_cls}' mistaken for '{pred_cls}'"
        )

    # ── Hardest correct predictions ───────────────────────────────────
    _header("Five hardest CORRECT predictions  (highest loss despite correct answer)")
    correct_idx = np.where(correct)[0]
    hard_order = correct_idx[losses[correct_idx].argsort()[::-1]][:5]

    for rank, idx in enumerate(hard_order):
        conf = P[idx, y_pred[idx]]
        print(
            f"\n  Rank {rank+1} | index={idx}  true={y_test[idx]}  "
            f"pred={y_pred[idx]}  confidence={conf:.2%}  loss={losses[idx]:.3f}"
        )
        digit_art = _ascii_digit(X_test[idx])
        for line in digit_art.split("\n")[::2]:  # half resolution for brevity
            print("    " + line)

    # ── Failure cases ─────────────────────────────────────────────────
    _header("Five WRONG predictions  (highest confidence on incorrect answer)")
    wrong_idx = np.where(~correct)[0]
    conf_wrong = P[wrong_idx, y_pred[wrong_idx]]
    worst_order = wrong_idx[conf_wrong.argsort()[::-1]][:5]

    for rank, idx in enumerate(worst_order):
        conf = P[idx, y_pred[idx]]
        print(
            f"\n  Rank {rank+1} | index={idx}  true={y_test[idx]}  "
            f"pred={y_pred[idx]}  confidence={conf:.2%}"
        )
        digit_art = _ascii_digit(X_test[idx])
        for line in digit_art.split("\n")[::2]:
            print("    " + line)

    # ── Final banner ──────────────────────────────────────────────────
    _header("Summary")
    print(f"  Model       : 784 → 256 → 128 → 10  (NumPy only)")
    print(f"  Parameters  : 235,146")
    print(f"  Test images : {len(y_test):,}")
    print(f"  Accuracy    : {acc:.4%}  ({n_right:,} correct, {n_wrong:,} wrong)")
    print()


if __name__ == "__main__":
    evaluate()
