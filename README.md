# Neural Network From Scratch

> **Building a fully-connected neural network using only NumPy and mathematics.**
>
> No TensorFlow.
> No PyTorch.
> No automatic differentiation.
> No machine learning frameworks.
>
> Just Python, NumPy, and linear algebra.

---

## Overview

This project is an educational implementation of a feedforward neural network built completely from scratch.

The objective is **not** to compete with modern deep learning libraries, but to understand every computation that happens behind them.

Instead of treating neural networks as a black box, this project rebuilds every component manually:

- Dense (Linear) Layers
- Weight Initialization
- Biases
- Forward Propagation
- Activation Functions
- Softmax
- Cross Entropy Loss
- Backpropagation
- Gradient Descent
- Mini-batch Training
- Prediction & Evaluation

By the end of the project, the network classifies handwritten digits from MNIST at **98.16% accuracy**.

---

## Motivation

Frameworks such as PyTorch and TensorFlow hide thousands of mathematical operations behind a few lines of code.

```python
loss.backward()
optimizer.step()
```

These two lines perform matrix differentiation, chain rule application, gradient computation, and parameter updates — without the user ever seeing how.

This project removes those abstractions.

Every multiplication. Every derivative. Every gradient. Every parameter update — implemented manually.

> **"What actually happens when a neural network learns?"**

This project answers that question from first principles.

---

## Project Structure

``` fileStructure
NeuralNetwork/
│
├── datasets/
│   └── mnist/
│
├── nn/
│   ├── layers.py        ← Dense layer, He init, forward + backward
│   ├── activations.py   ← ReLU, Softmax
│   ├── losses.py        ← Cross Entropy + fused Softmax backward
│   ├── optimizer.py     ← SGD with optional momentum
│   ├── network.py       ← Network class, train_step, predict, accuracy
│   └── utils.py         ← MNIST loader, one-hot encoding
│
├── train.py             ← Mini-batch training loop, weight saving
├── evaluate.py          ← Full test-set report, confusion matrix
├── predict.py           ← Single-image inference
│
├── README.md
└── requirements.txt
```

---

## Architecture

``` text
Input  (784)
  │
Dense  784 → 256    He initialization
  │
ReLU
  │
Dense  256 → 128    He initialization
  │
ReLU
  │
Dense  128 → 10     He initialization
  │
Softmax
  │
Prediction
```

Total trainable parameters: **235,146**

---

## Mathematics

### Single neuron

$$z = w_1 x_1 + w_2 x_2 + w_3 x_3 + b$$

### Full layer (matrix form)

$$Z = XW + b$$

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

### Softmax

$$P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Numerically stable implementation subtracts the row maximum before exponentiating.

### Cross Entropy Loss

$$L = -\frac{1}{N} \sum_n \sum_c y_{n,c} \log \hat{y}_{n,c}$$

### Fused Softmax + Cross Entropy Gradient

$$\frac{\partial L}{\partial Z} = \frac{P - Y}{N}$$

The Softmax Jacobian cancels analytically, leaving a clean difference between predictions and labels.

### SGD Update Rule

$$W \leftarrow W - \eta \cdot \nabla W$$

---

## Implementation Notes

### He Initialization

```python
scale = np.sqrt(2.0 / in_features)
W = np.random.randn(in_features, out_features) * scale
```

Keeps activation variance stable through ReLU layers. Without it, signals vanish or explode in deeper networks.

### Normalisation Convention

`CrossEntropyLoss.backward()` returns `(P - Y) / N` — the gradient is pre-normalised by batch size. `Dense.backward()` receives this and does **not** divide again. Dividing twice produces effective learning rate of `lr / N`, killing convergence.

This bug was caught during development by a numerical gradient check, which confirmed the analytical gradient was exactly `1/N` of the finite-difference estimate.

### Softmax Skip in Backward Pass

```python
# layers[:-1] skips Softmax — its gradient is fused into loss.backward()
for layer in reversed(self.layers[:-1]):
    grad = layer.backward(grad)
```

Calling `Softmax.backward()` separately would apply the transformation twice.

---

## Usage

```bash
pip install numpy

# Train (downloads MNIST automatically)
python train.py

# Evaluate on test set
python evaluate.py

# Predict a single image
python predict.py              # random image
python predict.py --index 42   # specific index
python predict.py --index 42 --show-all
```

---

## Hyperparameters

| Parameter     | Value |
|---------------|-------|
| Learning rate | 0.1   |
| Epochs        | 15    |
| Batch size    | 64    |
| Momentum      | 0.0   |
| Optimizer     | SGD   |

---

## Key Learning Outcomes

After completing this project you should understand:

- Why neural networks use matrices — computing all neurons simultaneously via `XW + b` instead of looping
- What weights represent — a learned linear map from input space to feature space
- Why biases exist — they shift the decision boundary off the origin
- Why activation functions are necessary — without them, any stack of linear layers collapses into one
- How Softmax converts logits into probabilities — and why it is numerically unstable without the max-subtraction trick
- Why Cross Entropy works — it penalises confident wrong answers with unbounded loss (`log(0) → ∞`)
- How gradients are calculated — chain rule applied layer by layer in reverse
- How the fused Softmax + CE gradient simplifies to `(P - Y) / N`
- How SGD updates parameters — and why the learning rate is the most sensitive hyperparameter
- Why mini-batches improve training — noise in gradients acts as regularisation, and shuffling prevents ordering bias

---

## Future Improvements

- Adam Optimizer
- Momentum SGD
- Batch Normalisation
- Dropout regularisation
- Learning Rate Scheduling
- L2 Weight Decay
- Convolutional layers (CNNs from scratch)
- Automatic Differentiation Engine
- Computational Graph Visualisation

---

## Requirements

``` text
numpy>=1.24.0
```

Nothing else.

---

## Educational Philosophy

The purpose of this project is **understanding, not abstraction**.

Every major operation — forward propagation, activation functions, loss computation, gradient calculation, and parameter updates — is written manually to expose the mechanics hidden behind high-level machine learning frameworks.

The most important moment in this project is the numerical gradient check. A bug was found where gradients were being divided by batch size twice — once in the loss backward pass and once in the Dense layer backward pass. The result was an effective learning rate of `0.1 / 64 ≈ 0.00156`, producing 92% accuracy instead of 98%. The fix was two lines. The lesson was irreplaceable.

By reconstructing these components from first principles, the project provides an intuitive foundation before transitioning to libraries such as PyTorch or TensorFlow.

---

## Results

Trained on 60,000 MNIST images. Evaluated on 10,000 unseen test images.

``` result
Best validation accuracy  : 98.16%
Final training loss       : 0.0070
Total parameters          : 235,146
```

### Training Curve

``` text
Epoch  1   loss=0.3074   train=95.38%   val=95.10%  ◀ best
Epoch  2   loss=0.1378   train=97.13%   val=96.71%  ◀ best
Epoch  3   loss=0.0955   train=97.87%   val=97.09%  ◀ best
Epoch  4   loss=0.0711   train=98.54%   val=97.49%  ◀ best
Epoch  5   loss=0.0565   train=98.80%   val=97.69%  ◀ best
Epoch  6   loss=0.0458   train=98.94%   val=97.51%
Epoch  7   loss=0.0378   train=99.06%   val=97.87%  ◀ best
Epoch  8   loss=0.0301   train=99.39%   val=97.98%  ◀ best
Epoch  9   loss=0.0240   train=99.59%   val=98.17%  ◀ best
Epoch 10   loss=0.0193   train=99.76%   val=98.08%
Epoch 11   loss=0.0157   train=99.81%   val=98.06%
Epoch 12   loss=0.0126   train=99.88%   val=98.22%  ◀ best
Epoch 13   loss=0.0104   train=99.91%   val=98.16%
Epoch 14   loss=0.0079   train=99.96%   val=98.36%  ◀ best
Epoch 15   loss=0.0062   train=99.91%   val=98.10%
```

### Per-Class Accuracy

| Digit | Accuracy |   | Digit | Accuracy |
|-------|----------|---|-------|----------|
| 0     | 99.08%   |   | 5     | 96.97%   |
| 1     | 99.21%   |   | 6     | 98.64%   |
| 2     | 97.77%   |   | 7     | 97.96%   |
| 3     | 98.22%   |   | 8     | 97.33%   |
| 4     | 98.88%   |   | 9     | 96.73%   |

### Most Confused Pairs

| True | Predicted | Count |                                                  Why                                                     |
|------|-----------|-------|----------------------------------------------------------------------------------------------------------|
| 9    | 4         | 12    |         Incomplete closure of the loop causes the digit to resemble the angular structure of a 4.        |
| 5    | 6         | 11    |               Strong lower curvature and a weak upper stroke produce a shape similar to 6.               |
| 9    | 7         |  9    |              Missing or faint loop leaves a dominant vertical/diagonal stroke resembling 7.              |
| 8    | 6         |  8    |                 Poor separation between the two loops makes 8 appear as a single-loop 6.                 |
| 7    | 2         |  8    |        Curved handwriting transforms the diagonal stroke into the characteristic upper curve of 2.       |
| 5    | 3         |  7    | Rounded writing style reduces the distinction between the vertical edge of 5 and the double-curve of 3.  |
| 5    | 9         |  7    |              Rounded top and closed lower curve can mimic the loop-and-tail structure of 9.              |
| 7    | 9         |  6    |                  Handwritten flourish creates a partial loop, causing confusion with 9.                  |
