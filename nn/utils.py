"""
utils.py
--------
Utility functions for the project.

Responsibilities
----------------
- Download the MNIST dataset if it is not already present.
- Parse the binary IDX file format that MNIST uses.
- Return train and test splits as NumPy arrays.
- Provide a helper to one-hot encode integer labels.

Nothing in this file is specific to the network architecture.
It is pure data plumbing.
"""

import os
import gzip
import struct
import urllib.request

import numpy as np

# ---------------------------------------------------------------------------
# MNIST source URLs (original Yann LeCun mirror via ossci-datasets on AWS)
# ---------------------------------------------------------------------------
MNIST_BASE_URL = (
    "https://raw.githubusercontent.com/" "golbin/TensorFlow-MNIST/master/mnist/data/"
)

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _download_mnist(save_dir: str) -> None:
    """
    Download every MNIST file into *save_dir* if it is not already there.

    Parameters
    ----------
    save_dir : str
        Directory where the raw .gz files will be stored.
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, filename in MNIST_FILES.items():
        dest = os.path.join(save_dir, filename)
        if os.path.exists(dest):
            continue

        url = MNIST_BASE_URL + filename
        print(f"Downloading {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")


# ---------------------------------------------------------------------------
# IDX parsers
# ---------------------------------------------------------------------------


def _parse_images(path: str) -> np.ndarray:
    """
    Parse an IDX3 image file (compressed or uncompressed).

    The IDX3 format header:
        4 bytes  magic number  (0x00 0x00 0x08 0x03)
        4 bytes  number of images
        4 bytes  number of rows
        4 bytes  number of columns
        N bytes  pixel data (uint8, 0-255)

    Parameters
    ----------
    path : str
        Path to a .gz or raw IDX3 file.

    Returns
    -------
    np.ndarray of shape (N, 784), dtype float32, values in [0, 1].
    """
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 0x00000803, f"Bad magic number in image file: {magic:#010x}"

        data = np.frombuffer(f.read(), dtype=np.uint8)

    # Reshape to (N, 784) and normalise to [0, 1]
    images = data.reshape(n_images, rows * cols).astype(np.float32) / 255.0
    return images


def _parse_labels(path: str) -> np.ndarray:
    """
    Parse an IDX1 label file (compressed or uncompressed).

    The IDX1 format header:
        4 bytes  magic number  (0x00 0x00 0x08 0x01)
        4 bytes  number of items
        N bytes  label data (uint8, 0-9)

    Parameters
    ----------
    path : str
        Path to a .gz or raw IDX1 file.

    Returns
    -------
    np.ndarray of shape (N,), dtype int64.
    """
    opener = gzip.open if path.endswith(".gz") else open

    with opener(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        assert magic == 0x00000801, f"Bad magic number in label file: {magic:#010x}"

        labels = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)

    return labels


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_mnist(data_dir: str = "dataset/mnist"):
    """
    Load the MNIST dataset, downloading it first if necessary.

    Parameters
    ----------
    data_dir : str
        Directory where the raw MNIST .gz files are stored (or will be saved).

    Returns
    -------
    X_train : np.ndarray, shape (60000, 784), float32 in [0, 1]
    y_train : np.ndarray, shape (60000,),     int64
    X_test  : np.ndarray, shape (10000, 784), float32 in [0, 1]
    y_test  : np.ndarray, shape (10000,),     int64
    """
    _download_mnist(data_dir)

    X_train = _parse_images(os.path.join(data_dir, MNIST_FILES["train_images"]))
    y_train = _parse_labels(os.path.join(data_dir, MNIST_FILES["train_labels"]))
    X_test = _parse_images(os.path.join(data_dir, MNIST_FILES["test_images"]))
    y_test = _parse_labels(os.path.join(data_dir, MNIST_FILES["test_labels"]))

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# One-hot encoding
# ---------------------------------------------------------------------------


def one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert an integer label vector into a one-hot matrix.

    Parameters
    ----------
    labels      : np.ndarray, shape (N,), integer class indices.
    num_classes : int, number of classes (default 10 for MNIST).

    Returns
    -------
    np.ndarray of shape (N, num_classes), dtype float32.
    Each row has exactly one 1.0 at the column matching the class index.
    """
    N = labels.shape[0]
    out = np.zeros((N, num_classes), dtype=np.float32)
    out[np.arange(N), labels] = 1.0
    return out
