# Neural Network From Scratch

> **A complete, working neural network built using only NumPy and mathematics.**
>
> No TensorFlow. No PyTorch. No automatic differentiation. No magic.
>
> Just Python, NumPy, and the mathematics that every deep learning framework hides from you.

This README is a standalone guide. You do not need to know anything about neural networks before reading it. By the end you will understand every line of code in this project. Not just what it does, but why it works and what would happen if you changed it.

---

## Results

Before the theory, here is what this project achieves.

The network classifies handwritten digits from the MNIST dataset.

``` text
Architecture  :  784 → 256 → 128 → 10
Parameters    :  235,146
Framework     :  NumPy only
```

Training progress over 15 epochs:

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

**Best validation accuracy: 98.15% on 10,000 unseen images.**

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Quick Start](#quick-start)
3. [The Problem We Are Solving](#the-problem-we-are-solving)
4. [Part 1 - The Single Neuron](#part-1-the-single-neuron)
5. [Part 2 - From Neuron to Layer](#part-2-from-neuron-to-layer)
6. [Part 3 - Why Linear Layers Alone Fail](#part-3-why-linear-layers-alone-fail)
7. [Part 4 - Activation Functions](#part-4-activation-functions)
8. [Part 5 - The Output Layer and Softmax](#part-5-the-output-layer-and-softmax)
9. [Part 6 - Measuring Error with Cross Entropy Loss](#part-6-measuring-error-with-cross-entropy-loss)
10. [Part 7 - Forward Propagation](#part-7-forward-propagation)
11. [Part 8 - How Learning Happens](#part-8-how-learning-happens)
12. [Part 9 - Backpropagation](#part-9-backpropagation)
13. [Part 10 - The SGD Optimizer](#part-10-the-sgd-optimizer)
14. [Part 11 - Weight Initialisation](#part-11-weight-initialisation)
15. [Part 12 - Mini-Batch Training](#part-12-mini-batch-training)
16. [Part 13 - Verifying Correctness](#part-13-verifying-correctness)
17. [Part 14 - A Bug, a Lesson, and the Fix](#part-14-a-bug-a-lesson-and-the-fix)
18. [Implementation Map](#implementation-map)
19. [Hyperparameters](#hyperparameters)
20. [Future Improvements](#future-improvements)

---

## Project Structure

``` file
NeuralNetworkFromScratch/
│
├── nn/                     core library
│   ├── layers.py           Dense layer - forward and backward
│   ├── activations.py      ReLU and Softmax
│   ├── losses.py           Cross Entropy loss
│   ├── optimizer.py        SGD with optional momentum
│   ├── network.py          Network class - assembles everything
│   └── utils.py            MNIST loader and one-hot encoder
│
├── train.py                training loop - run this to train
├── evaluate.py             full test-set evaluation and confusion matrix
├── predict.py              single-image inference
├── verify.py               gradient correctness check (run this after any backward() change)
│
├── dataset/mnist/          downloaded automatically on first run
├── saved_model/            weights saved here after training
│
├── requirements.txt
└── README.md
```

`verify.py` is the most important maintenance tool in the project. If you modify any backward pass, run it immediately to confirm the gradients are still mathematically correct. It compares analytical gradients from `backward()` against numerical estimates from finite differences and reports the relative error for every parameter.

---

## Quick Start

```bash
# Install the only dependency
pip install numpy

# Train (downloads MNIST automatically on first run)
python train.py

# Evaluate on the 10,000 test images
python evaluate.py

# Predict a single image
python predict.py
python predict.py --index 42
python predict.py --index 42 --show-all

# Verify backpropagation is mathematically correct
python verify.py
```

---

## The Problem We Are Solving

The MNIST dataset contains 70,000 images of handwritten digits, each 28 × 28 pixels in greyscale.

``` flowgraph
 ___
|   |        ┌─────────────────────────┐
| 5 |  --->  │ 784 pixel values (0-1)  │  ---> predict: 5
|___|        └─────────────────────────┘
```

Each image is a 28 × 28 grid of pixel values between 0 (black) and 1 (white). We flatten it into a vector of 784 numbers.

The task: look at those 784 numbers and output the correct digit (0 through 9).

This is a classification problem. The network learns a function:

``` function
f : R^784  →  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

We never hand-code what makes a 5 look like a 5. The network learns those rules itself from 60,000 labelled examples.

---

## Part 1: The Single Neuron

Everything in a neural network starts with one neuron.

A neuron takes several inputs, multiplies each by a weight, sums the results, adds a bias term, and produces one output.

``` flowgraph
        w₁
x₁ ─────────┐
        w₂  │
x₂ ─────────┼────────────►  z = w₁x₁ + w₂x₂ + w₃x₃ + b  ──►  output
        w₃  │  b (bias)
x₃ ─────────┘
```

In mathematics:

``` mathematics
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Or more compactly using dot product notation:

``` mathematics
z = w · x + b
```

### What are weights?

Each weight `wᵢ` controls how much influence input `xᵢ` has on the output. A large positive weight means this input strongly pushes the output up. A large negative weight means it pushes the output down. A weight near zero means this input barely matters.

Training a neural network is the process of finding the right values for all the weights.

### Why bias?

Without a bias term:

``` mathematics
z = w · x
```

This forces the output to be zero whenever the input is zero. Bias removes this constraint:

``` mathematics
z = w · x + b
```

With bias, the neuron can represent any linear relationship, not just those passing through the origin. This dramatically increases expressive power.

---

## Part 2: From Neuron to Layer

A single neuron produces one number. A layer produces many numbers simultaneously.

A layer with `out_features` neurons, each receiving `in_features` inputs, can be computed all at once using matrix multiplication.

### The matrix form

Instead of:

``` mathematics
Neuron 1: z₁ = w₁₁x₁ + w₁₂x₂ + ... + b₁
Neuron 2: z₂ = w₂₁x₁ + w₂₂x₂ + ... + b₂
Neuron 3: z₃ = w₃₁x₁ + w₃₂x₂ + ... + b₃
```

We write:

``` mathematics
Z = X @ W + b
```

Where:

``` text
X   shape (batch_size, in_features)    ← input matrix
W   shape (in_features, out_features)  ← weight matrix
b   shape (out_features,)              ← bias vector
Z   shape (batch_size, out_features)   ← output matrix
```

Every row of X is one training example. Every column of W is one neuron's weights. The matrix multiplication computes all neurons for all examples simultaneously.

### Why matrix multiplication?

For our first hidden layer:

``` text
X: batch of examples × 784 inputs
W: 784 inputs × 256 neurons
```

Computing this with nested loops would require millions of individual multiplications. With `X @ W`, NumPy delegates to optimised BLAS routines that use SIMD instructions and cache-aware algorithms. The mathematics is identical. The execution is orders of magnitude faster.

### In code: `nn/layers.py`

```python
class Dense:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)          # He initialisation
        self.W = np.random.randn(in_features, out_features) * scale
        self.b = np.zeros(out_features)

    def forward(self, X):
        self._X = X                                  # cache for backward()
        return X @ self.W + self.b
```

---

## Part 3: Why Linear Layers Alone Fail

Suppose we stack three Dense layers with no activations:

``` mathematics
Z₁ = X  @ W₁ + b₁
Z₂ = Z₁ @ W₂ + b₂
Z₃ = Z₂ @ W₃ + b₃
```

Substituting:

``` mathematics
Z₃ = X @ (W₁ @ W₂ @ W₃) + (some combined bias)
   = X @ W_combined + b_combined
```

No matter how many linear layers you stack, the composition is still a single linear transformation. A 100-layer network without activations is mathematically identical to a network with one layer.

A linear function can only draw straight lines (or hyperplanes in higher dimensions). It cannot learn non-linear decision boundaries, curved patterns, or hierarchical features.

This is the fundamental limitation that activation functions solve.

---

## Part 4: Activation Functions

An activation function is a non-linear function applied element-wise after each Dense layer. It breaks the linear composition and enables networks to approximate arbitrarily complex functions.

### ReLU (Rectified Linear Unit)

``` mathematics
ReLU(x) = max(0, x)
```

Every negative value becomes zero. Every positive value passes through unchanged.

``` text
Input : [-3,  2, -1,  0,  5, -0.5]
Output: [ 0,  2,  0,  0,  5,  0  ]
```

### Why ReLU and not sigmoid?

Sigmoid maps everything to (0, 1):

``` mathematics
σ(x) = 1 / (1 + e^(-x))
```

For large positive or large negative inputs, sigmoid saturates. Its gradient approaches zero. During backpropagation, repeatedly multiplying by near-zero gradients causes the signal to vanish before reaching early layers. This is the vanishing gradient problem.

ReLU has gradient exactly 1 for all positive inputs. It never saturates in the positive region. This is why deep networks use ReLU rather than sigmoid.

### The dead neuron problem

If a neuron's pre-activation value is always negative across all training examples, its gradient is always zero and it never updates. This is a dead neuron. He initialisation (Part 11) and careful learning rate selection reduce the likelihood of this happening.

### In code: `nn/activations.py`

```python
class ReLU:
    def forward(self, X):
        self._mask = X > 0             # cache: True where neuron is active
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * self._mask         # gradient flows only through active neurons
```

---

## Part 5: The Output Layer and Softmax

The final Dense layer produces 10 numbers, one per digit class. These are called logits.

``` text
Dense(128 → 10) output example:

  digit 0: -1.2
  digit 1:  0.4
  digit 2:  3.1   ← largest logit
  digit 3: -0.7
  ...
  digit 9:  1.8
```

Logits are raw scores. They are not probabilities. They can be negative, they do not sum to 1, and their scale is arbitrary. To convert them into a proper probability distribution, we apply Softmax.

### Softmax

``` mathematics
P_i = exp(z_i) / Σⱼ exp(z_j)
```

For each class, divide its exponentiated logit by the sum of all exponentiated logits.

Properties:

- Every output is strictly between 0 and 1
- All outputs sum to exactly 1
- The class with the highest logit always gets the highest probability

Example:

``` text
Logits:       [-1.2,  0.4,  3.1, -0.7, ...,  1.8]
After exp:    [ 0.3,  1.5, 22.2,  0.5, ...,  6.0]
Sum of exp:   38.5
Softmax:      [ 0.008, 0.039, 0.577, 0.013, ..., 0.156]
                                ↑
                   digit 2 wins with 57.7% confidence
```

### Numerical stability

`exp(1000)` overflows to infinity. We subtract the maximum logit before exponentiating:

```python
shift = Z - Z.max(axis=1, keepdims=True)   # max logit becomes 0
exp_z = np.exp(shift)                       # all exponents ≤ 0, no overflow
P     = exp_z / exp_z.sum(axis=1, keepdims=True)
```

This is mathematically identical to unshifted Softmax (the constant cancels in the division) but numerically safe for any input value.

### In code:  `nn/activations.py`

```python
class Softmax:
    def forward(self, Z):
        shift = Z - Z.max(axis=1, keepdims=True)
        exp_z = np.exp(shift)
        P = exp_z / exp_z.sum(axis=1, keepdims=True)
        self._P = P
        return P
```

---

## Part 6: Measuring Error with Cross Entropy Loss

After the network produces probabilities, we need a single number that answers: how wrong was this prediction? That number is the loss.

### Cross Entropy

``` mathematics
L = -(1/N) * Σₙ Σ_c  Y[n,c] * log(P[n,c])
```

Where N is the batch size, Y[n,c] is 1 if example n has true class c (else 0), and P[n,c] is the predicted probability of class c.

Because Y is one-hot, this reduces to:

``` mathematics
L = -(1/N) * Σₙ  log(P[n, true_class_n])
```

The loss is the negative mean log-probability of the correct class.

### Intuition

``` text
Correct AND confident:   P = 0.99  →  -log(0.99) ≈ 0.01   tiny loss
Correct but uncertain:   P = 0.60  →  -log(0.60) ≈ 0.51   moderate
Wrong AND confident:     P = 0.01  →  -log(0.01) ≈ 4.61   huge loss
```

Cross entropy punishes confident wrong answers severely. This encourages the network to be appropriately uncertain rather than confidently wrong.

### Why not Mean Squared Error?

For classification, MSE produces very small gradients when the network is confidently wrong, because the squared error function is flat near 0 and 1. Cross entropy has no such problem, its gradient approaches infinity as the predicted probability for the true class approaches zero.

### In code: `nn/losses.py`

```python
class CrossEntropyLoss:
    def forward(self, P, Y):
        N = P.shape[0]
        self._P, self._Y, self._N = P, Y, N
        return float(-np.sum(Y * np.log(np.clip(P, 1e-12, 1.0))) / N)
```

The clip to `[1e-12, 1.0]` prevents `log(0)` from producing negative infinity due to floating-point underflow.

---

## Part 7: Forward Propagation

The forward pass runs data through every layer in order and produces a prediction.

``` flowgraph
Input X                      shape (batch, 784)
    │
    ▼
Dense(784 → 256)             Z₁ = X @ W₁ + b₁
    │
    ▼
ReLU                         A₁ = max(0, Z₁)
    │
    ▼
Dense(256 → 128)             Z₂ = A₁ @ W₂ + b₂
    │
    ▼
ReLU                         A₂ = max(0, Z₂)
    │
    ▼
Dense(128 → 10)              Z₃ = A₂ @ W₃ + b₃
    │
    ▼
Softmax                      P  = softmax(Z₃)
    │
    ▼
CrossEntropyLoss             L  = -mean(Y * log(P))
    │
    ▼
Scalar loss
```

Each layer transforms the data into a progressively more abstract representation. By the final Dense layer, the 128 numbers going in are a learned encoding of the image. The 10 numbers coming out are scores for each digit class.

### The random baseline

Before any training, weights are random. Softmax over random weights is approximately uniform: each of 10 classes gets roughly 10% probability. The cross entropy of a uniform distribution over 10 classes is:

``` mathematics
-log(1/10) = log(10) ≈ 2.303
```

A first-epoch starting loss of approximately 2.3 confirms the forward pass is computing correctly. Any significantly different value suggests a bug in initialisation or the forward pass itself.

---

## Part 8: How Learning Happens

Forward propagation produces predictions, but predictions alone do not improve the model. The network has no idea why it was wrong or which weights caused the mistake.

Learning requires answering: for each weight, if I increase it slightly, does the loss go up or down? By how much?

This answer is the gradient of the loss with respect to each weight. Computing it efficiently for every weight in the network is the job of backpropagation.

### The gradient intuitively

Imagine the loss as a landscape and your position as the current values of all weights. You want to reach the lowest point in the landscape (minimum loss). The gradient tells you the slope at your current position. Taking a step in the opposite direction of the gradient moves you downhill toward lower loss.

---

## Part 9: Backpropagation

Backpropagation computes `∂L/∂W` for every weight in the network. It is a direct application of the chain rule from calculus, applied in reverse through the computation graph.

### The chain rule

If L depends on Z, and Z depends on W, then:

``` mathematics
∂L/∂W = ∂L/∂Z * ∂Z/∂W
```

For a chain of functions, the rule extends across every layer. Each layer computes its local gradient (how its output changes with its input) and passes the signal backward.

### The backward chain

``` text
Forward direction:                    Backward direction:

X → Dense₁ → ReLU → Dense₂ → ReLU → Dense₃ → Softmax → Loss

Loss.backward()      →  dZ₃ = (P - Y) / N
Dense₃.backward(dZ₃) →  dA₂, stores dW₃, db₃
ReLU₂.backward(dA₂)  →  dZ₂  (gates through stored mask)
Dense₂.backward(dZ₂) →  dA₁, stores dW₂, db₂
ReLU₁.backward(dA₁)  →  dZ₁  (gates through stored mask)
Dense₁.backward(dZ₁) →  dX,  stores dW₁, db₁
```

### Gradient through a Dense layer

For `Z = X @ W + b`:

``` text
∂L/∂W = X.T @ dZ               shape (in_features, out_features)
∂L/∂b = dZ.sum(axis=0)         shape (out_features,)
∂L/∂X = dZ @ W.T               shape (batch_size, in_features)
```

`∂L/∂X` is passed backward to the previous layer. `∂L/∂W` and `∂L/∂b` are stored on the layer object for the optimizer to use.

### Gradient through ReLU

ReLU's derivative is 1 where its input was positive, 0 elsewhere. The binary mask stored during `forward()` is applied directly:

``` mathematics
∂L/∂X = ∂L/∂A * (X > 0)
```

Gradients are blocked through neurons that were inactive (negative input). This is the ReLU gate.

### The fused Softmax + Cross Entropy gradient

Computing the Softmax gradient alone requires a full Jacobian, which is complex to implement and expensive to compute. But Softmax and Cross Entropy are always used together in classification and their combined gradient simplifies dramatically.

**Full derivation:**

The loss: `L = -(1/N) * Σₙ Σ_c  Y[n,c] * log(P[n,c])`

Gradient of L with respect to P[n,c]:

``` mathematics
∂L/∂P[n,c] = -(1/N) * Y[n,c] / P[n,c]
```

Softmax Jacobian (how P_i changes with logit Z_j):

``` mathematics
∂P_i/∂Z_j = P_i * (δᵢⱼ - P_j)
```

Applying the chain rule and summing over all Jacobian rows:

``` mathematics
∂L/∂Z[n,c] = Σᵢ  ∂L/∂P[n,i] * ∂P[n,i]/∂Z[n,c]

           = Σᵢ  [-(1/N) * Y[n,i]/P[n,i]] * P[n,i] * (δᵢ_c - P[n,c])

           = -(1/N) * Σᵢ  Y[n,i] * (δᵢ_c - P[n,c])

           = -(1/N) * [Y[n,c] - P[n,c] * Σᵢ Y[n,i]]
```

Since Y is one-hot, `Σᵢ Y[n,i] = 1` for every example:

``` mathematics
∂L/∂Z[n,c] = -(1/N) * (Y[n,c] - P[n,c])
            =  (1/N) * (P[n,c] - Y[n,c])
```

In matrix form:

``` mathematics
∂L/∂Z = (P - Y) / N
```

The entire Softmax Jacobian collapsed. The gradient is just the difference between predictions and labels, divided by batch size. This elegant result is why Softmax and Cross Entropy are always paired in classification networks.

### In code: `nn/losses.py` and `nn/layers.py`

```python
# losses.py
def backward(self):
    return (self._P - self._Y) / self._N

# layers.py
def backward(self, dZ):
    self.dW = self._X.T @ dZ     # gradient w.r.t. weights
    self.db = dZ.sum(axis=0)     # gradient w.r.t. biases
    return dZ @ self.W.T         # gradient w.r.t. inputs → passed to previous layer
```

### Normalisation convention - critical detail

`CrossEntropyLoss.backward()` returns `(P - Y) / N` - the gradient is already divided by the batch size N. Every layer in the backward chain receives a gradient that is already 1/N-scaled.

Therefore `Dense.backward()` uses `sum` (not `mean`) and does not divide by batch size again. Dividing twice produces gradients that are `1/N` of the correct value. See Part 14 for what happens when you make this mistake.

---

## Part 10: The SGD Optimizer

Once backpropagation has computed `∂L/∂W` for every parameter, the optimizer applies the updates.

### Vanilla SGD

``` mathematics
W ← W - η * ∂L/∂W
b ← b - η * ∂L/∂b
```

The learning rate η is the most sensitive hyperparameter:

``` text
η too large   →  loss oscillates or diverges
η too small   →  training crawls, converges slowly
η just right  →  steady, rapid descent
```

### SGD with Momentum

Momentum accumulates a velocity vector in the direction of consistent gradient:

``` mathematics
v ← μ * v + η * ∂L/∂W
W ← W - v
```

Where μ is typically 0.9. Momentum accelerates through consistent directions and dampens oscillations across conflicting ones.

### The training cycle

Every iteration follows this exact order:

``` alogorithm
optimizer.zero_grad()      ← clear gradients from previous step
P = net.forward(X)         ← forward pass
L = loss_fn.forward(P, Y)  ← compute loss
dZ = loss_fn.backward()    ← start backward chain
net._backward(dZ)          ← propagate through all layers
optimizer.step()           ← update weights
```

`zero_grad()` sets `dW = None`. If `step()` is called without a preceding `backward()`, it raises an assertion error immediately rather than silently using stale gradients.

### In code: `nn/optimizer.py`

```python
class SGD:
    def step(self):
        for layer in self.layers:
            layer.W -= self.lr * layer.dW
            layer.b -= self.lr * layer.db

    def zero_grad(self):
        for layer in self.layers:
            layer.dW = None
            layer.db = None
```

---

## Part 11: Weight Initialisation

The initial values of weights matter enormously.

### Why not all zeros?

If all weights are zero, every neuron in a layer computes the same output for any input. During backpropagation every neuron receives the same gradient and updates identically. The layer stays symmetric forever, effectively a single neuron no matter how wide it is. This is the symmetry problem.

### Why not large random values?

Large weights produce large pre-activation values. Softmax amplifies large differences, making the initial probabilities extreme before any training has happened. The gradients are also large and destabilise early training.

### He Initialisation

He initialisation (Kaiming He, 2015) scales random weights to keep activation variance stable through ReLU layers:

``` mathematics
W ~ Normal(0, sqrt(2 / in_features))
```

**Why `sqrt(2 / in_features)`:**

Suppose input activations have variance 1. After a linear layer with n inputs:

``` mathematics
Var(Z) = n * Var(w) * Var(x) = n * Var(w)
```

For `Var(Z) = 1` we need `Var(w) = 1/n`.

But ReLU zeros out roughly half the neurons, halving the effective variance. Compensating gives `Var(w) = 2/n`, or `std(w) = sqrt(2/n)`.

The factor of 2 is what distinguishes He from Xavier initialisation, which uses `1/n` and is designed for sigmoid/tanh, not ReLU.

### In code:  `nn/layers.py`

```python
scale = np.sqrt(2.0 / in_features)
self.W = np.random.randn(in_features, out_features) * scale
self.b = np.zeros(out_features)    # biases start at zero - no symmetry issue
```

---

## Part 12: Mini-Batch Training

There are three strategies for gradient descent:

**Batch gradient descent**: compute gradient over the entire training set before updating. Accurate gradient, but 60,000 forward passes per update step. Slow to start.

**Pure stochastic gradient descent**: one example per update. Many updates per epoch, but very noisy gradient, loss jumps wildly.

**Mini-batch SGD**: a small batch (typically 32–256 examples) per update. Efficient matrix operations, stable gradients, fast updates. Used universally in practice.

This project uses batches of 64.

### Why shuffle?

Without shuffling, the network sees the same sequence every epoch. It can overfit to the ordering of training data rather than the content. Shuffling before each epoch ensures each batch is a fresh random sample and prevents order-based overfitting.

### In code: `train.py`

```python
def iter_batches(X, Y, batch_size, shuffle=True):
    indices = np.arange(len(X))
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(X), batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], Y[idx]
```

---

## Part 13: Verifying Correctness

Writing backpropagation by hand is error-prone. A bug can produce a network that trains, converges, and reaches a plausible accuracy, just not the best possible one. Without a correctness test, the bug is invisible.

### Numerical gradient checking

The standard verification technique uses the definition of the derivative to estimate gradients without any chain rule:

``` mathematics
∂L/∂wᵢ ≈ ( L(wᵢ + h) - L(wᵢ - h) ) / (2h)
```

For each weight:

1. Increase it by a tiny amount h = 1e-5
2. Run a full forward pass and record the loss
3. Decrease it by h
4. Run another forward pass and record the loss
5. Estimate the gradient as (loss_plus - loss_minus) / (2h)

Compare this numerical estimate to the analytical gradient from `backward()`. If they agree to 4–5 significant figures, the backward pass is correct.

Results from `verify.py` on this project:

``` table
Parameter    Relative error     Status
──────────────────────────────────────
dense2.W     2.04e-11           ✓
dense2.b     4.03e-11           ✓
dense1.W     2.69e-11           ✓
dense1.b     3.72e-11           ✓
```

Errors in the `1e-11` range are machine-epsilon territory for float64. The gradients are exact.

```bash
python verify.py
```

`verify.py` uses a small network (8 → 6 → 4) so it completes in under a second despite requiring 2 × (parameter count) forward passes.

---

## Part 14: A Bug, a Lesson, and the Fix

During development, numerical gradient checking immediately flagged a problem.

Every analytical gradient was exactly `1/64` of the numerical estimate. Relative error: ~0.71. The `1/N` factor (for N = batch size = 64) is the exact signature of an off-by-N error.

The bug was in `Dense.backward()`:

```python
# WRONG - divides by batch_size when gradient is already normalised
self.dW = self._X.T @ dZ / batch_size
self.db = dZ.mean(axis=0)
```

`CrossEntropyLoss.backward()` already returns `(P - Y) / N`. Dividing again produced gradients that were `1/N²` of the correct value, making the effective learning rate `η / N = 0.1 / 64 ≈ 0.0016` instead of 0.1.

The result, after 15 epochs:

``` text
With bug    :  val accuracy = 92.58%
After fix   :  val accuracy = 98.15%
```

The fix was two lines:

```python
# CORRECT
self.dW = self._X.T @ dZ        # dZ is already (P-Y)/N - do not divide again
self.db = dZ.sum(axis=0)        # sum over batch, not mean
```

The loss curve with the bug looked normal. Training converged. The accuracy was plausible, a first-time builder would assume 92% was a reasonable result for this architecture. Only the gradient check proved the gradients were wrong.

This is the reason `verify.py` exists. Run it after any change to a backward pass.

---

## Implementation Map

Every concept in this README maps directly to a file and function:

|             Concept              |        File         |          Class / Function           |
|----------------------------------|---------------------|-------------------------------------|
| Dense layer, Z = XW + b          | `nn/layers.py`      | `Dense.forward()`                   |
| He initialisation                | `nn/layers.py`      | `Dense.__init__()`                  |
| Gradient w.r.t. W, b, X          | `nn/layers.py`      | `Dense.backward()`                  |
| ReLU and its gate mask           | `nn/activations.py` | `ReLU.forward()`, `ReLU.backward()` |
| Softmax with stability fix       | `nn/activations.py` | `Softmax.forward()`                 |
| Cross Entropy loss               | `nn/losses.py`      | `CrossEntropyLoss.forward()`        |
| Fused Softmax+CE gradient        | `nn/losses.py`      | `CrossEntropyLoss.backward()`       |
| SGD, momentum, zero_grad         | `nn/optimizer.py`   | `SGD`                               |
| Full forward pass                | `nn/network.py`     | `Network.forward()`                 |
| Full backward chain              | `nn/network.py`     | `Network._backward()`               |
| Training step                    | `nn/network.py`     | `Network.train_step()`              |
| MNIST download and parse         | `nn/utils.py`       | `load_mnist()`                      |
| One-hot encoding                 | `nn/utils.py`       | `one_hot()`                         |
| Mini-batch generator             | `train.py`          | `iter_batches()`                    |
| Weight saving / loading          | `train.py`          | `save_weights()`, `load_weights()`  |
| Training loop                    | `train.py`          | `train()`                           |
| Confusion matrix, error analysis | `evaluate.py`       | `evaluate()`                        |
| Single-image inference           | `predict.py`        | `predict()`                         |
| Numerical gradient check         | `verify.py`         | `run()`                             |

---

## Hyperparameters

|     Parameter   | Value |                           What changes if you adjust it                           |
|-----------------|-------|-----------------------------------------------------------------------------------|
| `LEARNING_RATE` | 0.1   | Too high → loss oscillates. Too low → slow convergence.                           |
| `EPOCHS`        | 15    | Val accuracy plateaus around epoch 14. More epochs reduce train loss but not val. |
| `BATCH_SIZE`    | 64    | Larger → more stable gradient but fewer updates per epoch.                        |
| `MOMENTUM`      | 0.0   | Set to 0.9 to enable momentum SGD. Try with a slightly lower learning rate.       |
| `HIDDEN_1`      | 256   | Wider → more capacity and parameters, slower per step.                            |
| `HIDDEN_2`      | 128   | Same.                                                                             |

---

## Per-Class Accuracy

| Digit | Accuracy | | Digit | Accuracy |
|-------|----------|-|-------|----------|
| 0     | 98.98%   | | 5     | 97.09%   |
| 1     | 99.12%   | | 6     | 98.33%   |
| 2     | 97.58%   | | 7     | 97.67%   |
| 3     | 98.61%   | | 8     | 97.43%   |
| 4     | 98.47%   | | 9     | 97.42%   |

Most confused pairs - all visually similar:

| True | Predicted | Count |                                                  Reason                                                  |
|------|-----------|-------|----------------------------------------------------------------------------------------------------------|
| 9    | 4         | 12    |         Incomplete closure of the loop causes the digit to resemble the angular structure of a 4.        |
| 5    | 6         | 11    |               Strong lower curvature and a weak upper stroke produce a shape similar to 6.               |
| 9    | 7         |  9    |              Missing or faint loop leaves a dominant vertical/diagonal stroke resembling 7.              |
| 8    | 6         |  8    |                 Poor separation between the two loops makes 8 appear as a single-loop 6.                 |
| 7    | 2         |  8    |        Curved handwriting transforms the diagonal stroke into the characteristic upper curve of 2.       |
| 5    | 3         |  7    | Rounded writing style reduces the distinction between the vertical edge of 5 and the double-curve of 3.  |
| 5    | 9         |  7    |              Rounded top and closed lower curve can mimic the loop-and-tail structure of 9.              |
| 7    | 9         |  6    |                  Handwritten flourish creates a partial loop, causing confusion with 9.                  |

---

## Future Improvements

The project keeps things deliberately simple. Natural extensions in approximate order of difficulty:

### Optimizers

- Adam: adaptive per-parameter learning rates (momentum + RMSProp combined)
- Learning rate scheduling: decay η after fixed epochs or on validation plateau

### Regularisation

- L2 weight decay: penalises large weights, reduces overfitting
- Dropout: randomly zeroes activations during training to prevent co-adaptation

### Network improvements

- Batch normalisation: normalises layer inputs, stabilises and accelerates training
- Convolutional layers: exploit spatial structure in images, reduce parameter count, improve accuracy

### Infrastructure

- Automatic differentiation: build a computation graph and derive gradients symbolically
- GPU support via CuPy (drop-in NumPy replacement on CUDA)

---

## Requirements

``` text
numpy>=1.24.0
```

Nothing else.

---

## Educational Philosophy

The purpose of this project is understanding, not abstraction.

Every framework call hides computation. `loss.backward()` in PyTorch runs thousands of operations - all invisible. Most people who use it have never derived the Softmax gradient, never seen the chain rule written out for a multi-layer network, and would not be able to explain why dividing by batch size in the wrong place loses 6 percentage points of accuracy.

This project makes all of it visible. Every derivative. Every matrix shape. Every edge case - the max-subtraction trick in Softmax, the clip in Cross Entropy, the reason the backward pass uses `sum` instead of `mean`, why the last layer (Softmax) is skipped in the backward chain.

The gradient check in `verify.py` exists because correctness cannot be assumed. A backward pass can look correct, produce a steadily decreasing loss, converge cleanly and still be wrong in a way that costs 6 percentage points. The only way to know is to compare the analytical gradient against the finite-difference estimate and verify that they agree to machine precision.

Build it from scratch once. You will never look at `loss.backward()` the same way again.
