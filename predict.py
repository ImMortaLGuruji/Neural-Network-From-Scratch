"""
predict.py
----------
Run inference on a single MNIST image and display the result.

Usage
-----
    python predict.py                        - picks a random test image
    python predict.py --index 42             - predicts image at index 42
    python predict.py --index 42 --show-all  - also show all 10 class scores

Requires
--------
    saved_model/weights.npz - produced by train.py
"""

import argparse
import numpy as np

from nn.utils import load_mnist
from train import build_network, load_weights, WEIGHTS_FILE, DATA_DIR

# ══════════════════════════════════════════════════════════════════════════
# Rendering helpers
# ══════════════════════════════════════════════════════════════════════════


def _ascii_image(pixels: np.ndarray, scale: int = 2) -> str:
    """
    Render a 784-vector as an ASCII image.

    Each pixel is mapped to a character from a brightness ramp.
    `scale=2` doubles the horizontal width to correct for monospace
    aspect ratio (characters are taller than they are wide).
    """
    ramp = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    image = pixels.reshape(28, 28)
    rows = []
    for row in image:
        line = ""
        for v in row:
            ch = ramp[int(v * (len(ramp) - 1))]
            line += ch * scale  # double width for aspect ratio
        rows.append(line)
    return "\n".join(rows)


def _confidence_bar(prob: float, width: int = 32) -> str:
    filled = int(prob * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {prob:>6.2%}"


# ══════════════════════════════════════════════════════════════════════════
# Main prediction
# ══════════════════════════════════════════════════════════════════════════


def predict(index: int | None = None, show_all: bool = False) -> None:

    W = 70

    def sep(c="─"):
        print(c * W)

    def header(t):
        print(f"\n{'═'*W}\n  {t}\n{'═'*W}")

    # ── Load data & model ─────────────────────────────────────────────
    _, _, X_test, y_test = load_mnist(DATA_DIR)
    net = build_network()
    load_weights(net, WEIGHTS_FILE)

    # ── Pick image ────────────────────────────────────────────────────
    if index is None:
        index = int(np.random.randint(0, len(X_test)))

    assert 0 <= index < len(X_test), f"Index {index} out of range [0, {len(X_test)-1}]"

    x = X_test[index]  # (784,)
    true = int(y_test[index])

    # ── Inference ─────────────────────────────────────────────────────
    P = net.forward(x.reshape(1, -1))[0]  # (10,)
    pred = int(P.argmax())
    conf = float(P[pred])
    result = "✓  CORRECT" if pred == true else "✗  WRONG"

    # ── Print ─────────────────────────────────────────────────────────
    header(f"Predicting MNIST test image  (index {index})")

    # ASCII image
    sep()
    for line in _ascii_image(x).split("\n"):
        print("  " + line)
    sep()

    # Verdict
    print(f"\n  True label   : {true}")
    print(f"  Predicted    : {pred}   {result}")
    print(f"  Confidence   : {_confidence_bar(conf)}")

    # All class scores
    if show_all:
        header("All class probabilities")
        sorted_idx = P.argsort()[::-1]
        for rank, cls in enumerate(sorted_idx):
            marker = (
                " ◀ predicted"
                if cls == pred
                else " ◀ true" if cls == true and cls != pred else ""
            )
            print(
                f"  Rank {rank+1}  digit {cls}  {_confidence_bar(float(P[cls]), 24)}{marker}"
            )

    # Top-3 guesses
    else:
        header("Top-3 guesses")
        top3 = P.argsort()[::-1][:3]
        for rank, cls in enumerate(top3):
            marker = "  ◀ true" if cls == true and cls != pred else ""
            print(
                f"  #{rank+1}  digit {cls}  {_confidence_bar(float(P[cls]), 24)}{marker}"
            )

    print()


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single MNIST image.")
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Test-set index (0-9999). Random if omitted.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print probabilities for all 10 classes.",
    )
    args = parser.parse_args()
    predict(index=args.index, show_all=args.show_all)
