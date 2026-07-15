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

* Dense (Linear) Layers
* Weight Initialization
* Biases
* Forward Propagation
* Activation Functions
* Softmax
* Cross Entropy Loss
* Backpropagation
* Gradient Descent
* Mini-batch Training
* Prediction & Evaluation

By the end of the project, the network is capable of learning and classifying handwritten digits from the MNIST dataset.

---

## Motivation

Frameworks such as PyTorch and TensorFlow hide thousands of mathematical operations behind a few lines of code.

```python
loss.backward()
optimizer.step()
```

These two lines perform:

* Matrix differentiation
* Chain rule
* Gradient computation
* Parameter updates

without the user ever seeing how.

This project removes those abstractions.

Every multiplication.

Every derivative.

Every gradient.

Every parameter update.

is implemented manually.

The goal is to answer one question:

> **"What actually happens when a neural network learns?"**

---

## Project Goals

* Learn neural networks mathematically
* Understand forward propagation
* Implement backpropagation manually
* Learn how gradients are calculated
* Build reusable neural network components
* Train a network without any ML framework
* Gain intuition behind deep learning

---

## Features

✔ Fully Connected Neural Network

✔ Modular Layer Design

✔ ReLU Activation

✔ Softmax Output

✔ Cross Entropy Loss

✔ Backpropagation

✔ Stochastic Gradient Descent (SGD)

✔ Mini-batch Training

✔ Accuracy Evaluation

✔ MNIST Classification

---

## Project Structure

``` fileStructure
NeuralNetwork/
│
├── dataset/
│   └── mnist/
│
├── nn/
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── network.py
│   └── utils.py
│
├── train.py
├── predict.py
├── evaluate.py
│
├── README.md
└── requirements.txt
```

---

## Neural Network Architecture

The network consists of multiple fully connected layers.

``` pipeline
Input Image
      │
      ▼
Linear Layer
      │
      ▼
ReLU
      │
      ▼
Linear Layer
      │
      ▼
ReLU
      │
      ▼
Linear Layer
      │
      ▼
Softmax
      │
      ▼
Prediction
```

---

## From One Neuron to Many

Everything begins with a single neuron.

A neuron receives several inputs

``` mathematics
x₁
x₂
x₃
```

Each input has its own weight

``` mathematics
w₁
w₂
w₃
```

The neuron computes

[
z=w_1x_1+w_2x_2+w_3x_3+b
]

where

* **w** = weights
* **x** = inputs
* **b** = bias

The output is simply a weighted sum.

---

### Why Bias?

Without a bias, every neuron is forced to pass through the origin.

Bias allows the neuron to shift its decision boundary.

Mathematically,

[
y = Wx+b
]

instead of

[
y = Wx
]

This dramatically increases the flexibility of the model.

---

## From Neurons to Layers

Instead of computing one neuron at a time, we compute all neurons simultaneously using matrix multiplication.

Instead of

``` mathematics
Neuron 1

Neuron 2

Neuron 3
```

we compute

[
Z=XW+B
]

where

* X = inputs
* W = weights
* B = biases

This is why neural networks heavily rely on linear algebra.

---

## Why Matrix Multiplication?

Imagine computing

``` mathematics
100 neurons

×

784 inputs
```

using nested loops.

That would require tens of thousands of multiplications.

Using

```python
np.dot(inputs, weights)
```

NumPy performs the same computation using highly optimized linear algebra routines.

---

## The Problem with Linear Layers

Suppose our network contains only linear layers.

``` mathematics
Linear

↓

Linear

↓

Linear
```

Mathematically,

[
A(B(Cx))
]

is still just another linear transformation.

Therefore,

multiple linear layers collapse into one.

The network cannot learn complex relationships.

---

## Introducing Non-Linearity

To solve this, activation functions are inserted after every layer.

This project uses

### ReLU

(Rectified Linear Unit)

[
ReLU(x)=\max(0,x)
]

It simply replaces all negative values with zero.

Example

``` mathematics
Input

[-3,2,-1,5]

↓

Output

[0,2,0,5]
```

ReLU enables neural networks to approximate highly complex functions.

---

## Output Layer

The final layer produces numbers called **logits**.

Example

``` mathematics
[2.3, 1.2, 7.5]
```

These are **not probabilities**.

To convert them into probabilities, we use Softmax.

---

## Softmax

Softmax transforms arbitrary values into probabilities.

[
P_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
]

Properties

* Every probability is positive.
* All probabilities sum to 1.
* Highest probability becomes the prediction.

Example

``` mathematics
Logits

[2,4,1]

↓

Softmax

[0.11
 0.82
 0.07]
```

---

## Loss Function

Once predictions are made, we compare them with the correct labels.

This project uses

### Cross Entropy Loss

[
L=-\sum y\log(\hat y)
]

Cross entropy measures

> "How wrong was the prediction?"

Smaller loss means better predictions.

---

## Forward Propagation

The forward pass is simply repeated application of

``` pipeline
Linear Layer

↓

Activation

↓

Linear Layer

↓

Activation

↓

Output
```

or mathematically

[
X
\rightarrow
WX+b
\rightarrow
ReLU
\rightarrow
WX+b
\rightarrow
Softmax
]

The result is a probability distribution over every class.

---

## How Does Learning Happen?

Forward propagation alone does not improve the model.

Learning begins after computing the loss.

The network asks

> Which weights caused this mistake?

To answer this, it computes gradients.

---

## Backpropagation

Backpropagation computes

[
\frac{\partial Loss}{\partial Weight}
]

for every parameter.

Using the chain rule,

the error is propagated backwards through every layer.

``` pipeline
Prediction

↓

Loss

↓

Output Layer

↓

Hidden Layer

↓

Input Layer
```

Every weight receives feedback indicating

* increase
* decrease
* by how much

---

## Gradient Descent

Once gradients are known,

weights are updated.

[
W=W-\eta \nabla W
]

where

* η = learning rate

This process repeats thousands of times.

``` pipeline
Forward

↓

Loss

↓

Backward

↓

Update

↓

Repeat
```

Eventually the network converges.

---

## Learning Rate

The learning rate controls the size of each update.

Too small

``` english
Learning is slow
```

Too large

``` english
Training becomes unstable
```

Choosing a good learning rate is one of the most important hyperparameters.

---

## Optimizer

This project implements

### Stochastic Gradient Descent (SGD)

Update rule

[
W=W-\eta\frac{\partial L}{\partial W}
]

Although simple, SGD remains one of the most fundamental optimization algorithms in deep learning.

---

## Mini-Batch Training

Instead of processing the entire dataset at once,

training data is divided into smaller batches.

Advantages

* Faster computation
* Better GPU/CPU utilization
* More stable gradients
* Reduced memory usage

---

## Training Pipeline

``` pipeline
Load Dataset

↓

Shuffle Data

↓

Create Mini Batches

↓

Forward Pass

↓

Loss

↓

Backward Pass

↓

Update Weights

↓

Repeat
```

---

## Evaluation

After training,

the model predicts unseen images.

Accuracy is computed as

[
Accuracy=
\frac{Correct}{Total}
]

Higher accuracy indicates better generalization.

---

## Dataset

### MNIST

* 60,000 training images
* 10,000 testing images
* 28 × 28 grayscale
* Digits 0–9

---

## Learning Outcomes

After completing this project you should understand

* Why neural networks use matrices
* What weights actually represent
* Why biases exist
* Why activation functions are necessary
* How Softmax converts logits into probabilities
* Why Cross Entropy works
* How gradients are calculated
* How the chain rule enables learning
* How SGD updates parameters
* Why mini-batches improve training
* What modern frameworks automate behind the scenes

---

## Future Improvements

Possible extensions include:

* Xavier Initialization
* He Initialization
* Adam Optimizer
* RMSProp
* Momentum SGD
* Batch Normalization
* Dropout
* Learning Rate Scheduling
* L2 Regularization
* CNNs from Scratch
* Automatic Differentiation Engine
* GPU Acceleration
* Computational Graph Visualization

---

## Educational Philosophy

The purpose of this project is **understanding, not abstraction**.

Every major operation—forward propagation, activation functions, loss computation, gradient calculation, and parameter updates—is written manually to expose the mechanics hidden behind high-level machine learning frameworks. By reconstructing these components from first principles, the project provides an intuitive understanding of how neural networks learn from data, making it an ideal foundation before transitioning to libraries such as PyTorch or TensorFlow.
