# this is a learners file for my peers 
# Complete Beginner's Guide to ReIQA: From PyTorch Basics to ResNet and Contrastive Learning

Welcome! This guide will take you from complete beginner to understanding your entire codebase. We'll learn step-by-step, starting with the absolute basics.

---

## Table of Contents

1. [Part 1: PyTorch Fundamentals](#part-1-pytorch-fundamentals)
2. [Part 2: Neural Networks & Convolutions](#part-2-neural-networks--convolutions)
3. [Part 3: ResNet Architecture](#part-3-resnet-architecture)
4. [Part 4: Contrastive Learning & MoCo](#part-4-contrastive-learning--moco)
5. [Part 5: Your ReIQA Codebase](#part-5-your-reiqa-codebase)
6. [Part 6: Complete Workflow](#part-6-complete-workflow)

---

# PART 1: PYTORCH FUNDAMENTALS

## What is PyTorch?

PyTorch is a **deep learning library** - think of it as a toolbox for building artificial brains that can learn from data.

### Why PyTorch?
- **Easy to learn** - Code reads like normal Python
- **GPU support** - Run things super fast on graphics cards
- **Dynamic graphs** - Change what your network does on the fly
- **Great for research** - Flexible and powerful

### Basic Building Block: TENSORS

A **tensor** is just a fancy word for a multi-dimensional array of numbers. Think of it like a container:

```python
import torch

# 0D tensor (just a number)
scalar = torch.tensor(5.0)
print(scalar)  # tensor(5.)

# 1D tensor (list of numbers)
vector = torch.tensor([1.0, 2.0, 3.0])
print(vector)  # tensor([1., 2., 3.])

# 2D tensor (matrix - like a spreadsheet)
matrix = torch.tensor([[1.0, 2.0],
                       [3.0, 4.0]])
print(matrix)
# tensor([[1., 2.],
#         [3., 4.]])

# 3D tensor (cube of numbers)
cube = torch.randn(2, 3, 4)  # 2 dimensions of 3x4
print(cube.shape)  # torch.Size([2, 3, 4])

# 4D tensor (what images are!)
# This represents 1 image with 3 color channels (RGB), 224x224 pixels
image_tensor = torch.randn(1, 3, 224, 224)
#               batch  channels  height  width
```

**Key Point**: Images are just 4D tensors!
- Dimension 1: How many images in this batch (usually 32, 64, etc.)
- Dimension 2: Color channels (3 for RGB: Red, Green, Blue)
- Dimension 3: Height in pixels
- Dimension 4: Width in pixels

### Basic Tensor Operations

```python
# Creating tensors
ones = torch.ones(3, 3)      # Matrix of all 1s
zeros = torch.zeros(3, 3)    # Matrix of all 0s
random_nums = torch.randn(3, 3)  # Random numbers

# Math operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

z = x + y          # Addition: [5., 7., 9.]
z = x * y          # Element-wise multiplication: [4., 10., 18.]
z = torch.matmul(x, y)  # Dot product (special multiplication)

# Reshaping
original = torch.randn(12)      # 12 numbers in a line
reshaped = original.reshape(3, 4)  # Reshape to 3x4
flattened = reshaped.flatten()   # Flatten back to line

# Getting values
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
value = tensor[0, 1]  # Get element at row 0, col 1 -> 2.0
row = tensor[0]       # Get entire first row -> [1., 2.]
```

### Moving to GPU (for speed!)

```python
# Most modern machines have GPUs (graphics cards that are great for math)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move tensor to GPU
x = torch.randn(1000, 1000)
x = x.to(device)  # Now x is on GPU and super fast!

# Or create directly on GPU
y = torch.randn(1000, 1000, device=device)
```

---

## Basic Neural Network Concepts

### What is a Neural Network?

A neural network is just a bunch of mathematical transformations stacked together:

```
Input (pixels) → Layer 1 → Layer 2 → Layer 3 → Output (prediction)
                 (learn)   (learn)   (learn)
```

Each layer learns to transform data into something more useful.

### Gradient Descent: How Networks Learn

Imagine you're lost in fog on a mountain and want to reach the valley (minimize loss):

```
1. You're at some position (random weights)
2. Look around and see which direction is downhill (compute gradient)
3. Take a small step downhill (update weights)
4. Repeat until you reach the bottom (minimum loss)
```

```python
import torch
import torch.nn as nn

# Simple learning example
x = torch.tensor([[1.0], [2.0], [3.0]])  # Input
y = torch.tensor([[2.0], [4.0], [6.0]])  # Target (y = 2*x)

# Create a simple model: single linear transformation
model = nn.Linear(1, 1)  # 1 input, 1 output
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr=learning rate (step size)
loss_fn = nn.MSELoss()  # Mean Squared Error loss

# Training loop
for epoch in range(100):
    # Forward pass (make prediction)
    predictions = model(x)

    # Calculate loss (how wrong we are)
    loss = loss_fn(predictions, y)

    # Backward pass (calculate gradients - "which direction to go?")
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Calculate new gradients

    # Update weights (take a step downhill)
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# The model learned that y = 2*x!
print(f"Learned weight: {model.weight.item()}")  # Should be ~2.0
```

### Key Terms Explained:

- **Epoch**: One full pass through all training data
- **Loss**: How wrong our predictions are (smaller = better)
- **Gradient**: Direction of steepest descent (where to update weights)
- **Learning rate (lr)**: How big of a step to take (0.01 is small and safe)
- **Optimizer**: Algorithm that decides how to update weights (SGD, Adam, etc.)

---

# PART 2: NEURAL NETWORKS & CONVOLUTIONS

## Fully Connected Layers (FC Layers)

The simplest building block:

```python
import torch.nn as nn

# A layer that transforms 100 inputs to 50 outputs
fc_layer = nn.Linear(100, 50)

x = torch.randn(32, 100)  # Batch of 32 samples, each with 100 values
y = fc_layer(x)           # Output: (32, 50)

# What's happening mathematically:
# y = x @ W^T + b
# where W is learned weights and b is learned bias
```

**Problem with FC layers for images**: An image with 3 channels, 224x224 pixels requires:
- Flattened size: 3 × 224 × 224 = 150,528 numbers
- If output is 4096 numbers, we need 150,528 × 4096 = 616 MILLION weights!
- This is slow, memory-heavy, and doesn't capture image structure

## Convolutional Layers: The Better Solution

Instead of connecting every input to every output, we use **small sliding windows** called **convolutional kernels or filters**.

### How Convolution Works (Simple Explanation)

Imagine a 3×3 filter sliding over an image:

```
Input image:          Filter (3x3):        Output:
1 2 3                 0.5 0.1 -0.3
2 4 1         ×      0.2 0.8  0.1    →
5 3 2                -0.1 0.0  0.5

Position (0,0): (1×0.5 + 2×0.1 + 2×0.2 + 4×0.8 + 3×0.1 + 5×-0.1 + 3×0.0 + 2×0.5) = 4.5
(and so on for each position...)
```

The filter slides across the entire image, and at each position, we compute a dot product.

### Why This is Better

```python
import torch.nn as nn

# Convolutional layer: input 3 channels, output 64 channels, 3x3 kernel
conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

# Input: batch of 32 images, 3 RGB channels, 224×224 pixels
x = torch.randn(32, 3, 224, 224)

# Output: batch of 32, 64 channels (filters learned features), 224×224
y = conv(x)
print(y.shape)  # torch.Size([32, 64, 224, 224])

# Number of parameters: only 3 × 3 × 3 × 64 = 1,728!
# (kernel_height × kernel_width × in_channels × out_channels)
# Much better than 150,528 × 4096!
```

**Key insight**: The filter doesn't change as it slides - same 3×3 numbers are used everywhere. The network learns what useful patterns these 3×3 numbers should be!

### Other Important Layers

```python
# Batch Normalization: Normalize layer outputs to be stable
# This makes training faster and more stable
bn = nn.BatchNorm2d(64)

# ReLU (Rectified Linear Unit): Simple but powerful activation
# f(x) = max(0, x)  (zero out negatives, keep positives)
relu = nn.ReLU(inplace=True)

# MaxPool: Reduce spatial dimensions by taking maximum
# Usually 2×2, so 224×224 → 112×112
maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

# Adaptive Average Pool: Reduce ANY size to a single number per channel
# Useful at end of network before classification
avgpool = nn.AdaptiveAvgPool2d((1, 1))
```

### Full Simple CNN Example

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Layer 1: Conv + ReLU
        # 3 input channels (RGB) → 32 output channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)

        # Layer 2: Conv + ReLU + MaxPool
        # 32 input channels → 64 output channels, then reduce spatial size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 3: FC layer for classification
        # After 2x MaxPool (224→112→56), and 64 channels: input is 64*56*56
        self.fc = nn.Linear(64 * 56 * 56, 10)  # 10 classes

    def forward(self, x):
        x = self.conv1(x)           # (B, 3, 224, 224) → (B, 32, 224, 224)
        x = self.relu1(x)           # Keep same shape, apply ReLU

        x = self.conv2(x)           # (B, 32, 224, 224) → (B, 64, 224, 224)
        x = self.relu2(x)           # Keep same shape
        x = self.pool(x)            # (B, 64, 224, 224) → (B, 64, 112, 112)

        x = x.flatten(1)            # (B, 64, 112, 112) → (B, 64*112*112)
        x = self.fc(x)              # (B, 64*112*112) → (B, 10)

        return x

# Try it out!
model = SimpleCNN()
image_batch = torch.randn(4, 3, 224, 224)  # 4 images
output = model(image_batch)
print(output.shape)  # torch.Size([4, 10])
```

---

# PART 3: RESNET ARCHITECTURE

## The Problem ResNet Solved

As neural networks got deeper (more layers), they started getting **worse** at learning, not better. Paradoxically, more layers = worse performance. Why?

**Vanishing Gradient Problem**: As gradients backpropagate through many layers, they multiply by numbers < 1, becoming tiny → weights don't update.

## The Clever Solution: Skip Connections (Residual Blocks)

Instead of just stacking layers, ResNet adds **shortcuts** that skip over blocks of layers:

```
Simple network:
Input → Linear → ReLU → Linear → ReLU → Output

ResNet (Residual Block):
Input ────────────┐
      │           │
      ▼ Conv      │
    ReLU          │
      ▼ Conv      │
    Batch Norm    │
      │           │
      ├──────────►┤ Add (element-wise)
                  ▼
                Output
```

**Key idea**: The network learns **residuals** (differences) instead of absolute values.

### It's Like This:

```
Traditional: output = F(input)
ResNet:      output = input + F(input)  ← Add the INPUT back!
```

If the layers don't need to do anything, they can learn F(input) = 0, and output = input (identity).

### Why This Matters:

1. **Gradient flows better**: Gradients bypass layers through shortcuts
2. **Sparser networks**: Each block only needs to learn differences
3. **Enables much deeper networks**: We can use 50, 101, 152 layers without vanishing gradients

---

## ResNet Architecture Explained (From Your Code)

Let's break down your `resnet.py` file:

### BasicBlock vs Bottleneck

```python
# From your resnet.py:

class BasicBlock(nn.Module):
    expansion = 1  # Output channels = input channels

    def __init__(self, inplanes, planes, stride=1, ...):
        super(BasicBlock, self).__init__()

        # Structure: Conv(3×3) → BN → ReLU → Conv(3×3) → BN
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample  # Shortcut connection
        self.stride = stride

    def forward(self, x):
        identity = x  # Save input for skip connection

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Add residual connection: output = layer_output + input
        if self.downsample is not None:
            identity = self.downsample(x)  # Adjust dimensions if needed

        out += identity  # Element-wise addition
        out = self.relu(out)

        return out
```

**Bottleneck** (used in ResNet-50, ResNet-101):

```python
class Bottleneck(nn.Module):
    expansion = 4  # Output channels = input channels × 4

    def forward(self, x):
        # Structure: Conv(1×1) → Conv(3×3) → Conv(1×1)
        # Reduces channels with 1×1, processes with 3×3, expands with 1×1

        out = self.conv1(x)      # 256 → 64 channels (1×1)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)    # 64 → 64 channels (3×3, expensive layer)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)    # 64 → 256 channels (1×1)
        out = self.bn3(out)

        # Skip connection
        if self.downsample is not None:
            x = self.downsample(x)

        out += x  # Add residual
        out = self.relu(out)

        return out
```

**Why Bottleneck?** The 1×1 convolutions are "cheap" (reduce computation), and the important 3×3 computation happens in fewer channels. It's more efficient!

### Full ResNet Construction

```python
class ResNet(nn.Module):
    def __init__(self, block, layers, ...):
        super(ResNet, self).__init__()

        # First layer: Large 7×7 convolution to extract initial features
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # Input: (B, 3, 224, 224) → Output: (B, 64, 112, 112)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Max pooling: reduce size more
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # (B, 64, 112, 112) → (B, 64, 56, 56)

        # Four stages of residual blocks
        # layers = [3, 4, 6, 3] means 3+4+6+3=16 blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # Many blocks operating at 56×56 spatial size

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # stride=2 means reduce spatial size: 56→28

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # Reduce size: 28→14

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # Reduce size: 14→7

        # Average pool and flatten
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # (B, 512*expansion, 7, 7) → (B, 512*expansion, 1, 1)

        # For classification (usually removed in unsupervised learning)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x):
        # x shape: (B, 3, 224, 224)

        x = self.conv1(x)        # (B, 3, 224, 224) → (B, 64, 112, 112)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)      # → (B, 64, 56, 56)

        x = self.layer1(x)       # → (B, 64*exp, 56, 56)
        x = self.layer2(x)       # → (B, 128*exp, 28, 28)
        x = self.layer3(x)       # → (B, 256*exp, 14, 14)
        x = self.layer4(x)       # → (B, 512*exp, 7, 7)

        x = self.avgpool(x)      # → (B, 512*exp, 1, 1)
        x = torch.flatten(x, 1)  # → (B, 512*exp)
        x = self.fc(x)           # → (B, num_classes)

        return x
```

### Different ResNet Variants

From your code:

```python
def resnet18(pretrained=False, progress=True, **kwargs):
    # layers = [2, 2, 2, 2] = 2+2+2+2 = 8 blocks
    # Total depth: 7 (conv1) + 8×2 (BasicBlocks) + 1 (fc) ≈ 18 layers
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], ...)

def resnet50(pretrained=False, progress=True, **kwargs):
    # layers = [3, 4, 6, 3] = 3+4+6+3 = 16 blocks
    # Each Bottleneck has 3 convs, so ~16×3 = 48 layers + initial/fc ≈ 50 layers
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], ...)

def resnet101(pretrained=False, progress=True, **kwargs):
    # layers = [3, 4, 23, 3] = 3+4+23+3 = 33 blocks → ~100 layers
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], ...)

def resnet152(pretrained=False, progress=True, **kwargs):
    # layers = [3, 8, 36, 3] = 3+8+36+3 = 50 blocks → ~150 layers
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], ...)
```

**Deeper = Better (Usually)**
- ResNet18: 18 layers, fast, less powerful
- ResNet50: 50 layers, good balance
- ResNet101: 101 layers, more powerful
- ResNet152: 152 layers, most powerful but slower

---

## Pre-trained Models

Your code loads pre-trained weights:

```python
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    # ...
}

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained:
        # Download weights learned on ImageNet (1.2 million images, 1000 classes)
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)  # Load these pre-trained weights

    return model
```

**Why pre-trained?**
- Training from scratch on 1.2M images took weeks on many GPUs
- We can download weights and fine-tune them (transfer learning)
- Often gives better results than training from scratch
- Much faster!

---

# PART 4: CONTRASTIVE LEARNING & MoCo

## Why Contrastive Learning?

Normally, training requires **labels** (e.g., "this is a dog").

**Contrastive learning** learns from **unlabeled data** by comparing images:

```
"These two images are similar" (same image, different augmentations)
"These two images are different" (different images)
```

The network learns features where:
- Similar images are close in learned space
- Different images are far apart

### Simple Intuition

```
Without labels:            With contrastive learning:
(Can't train)              img_1 ─┐
                                  ├─ SIMILAR ─→ [Encoder] ─→ Features are close
                           img_1' ┘

                           img_1 ──→ [Encoder] ─→ Different images
                           img_2 ──→ [Encoder] ─→ get different features
```

**Advantage**: You can use ANY unlabeled data! Great when labeled data is expensive.

---

## MoCo: Momentum Contrast

Your codebase uses **MoCo** (Momentum Contrast), a powerful contrastive learning method.

### Key Idea: Two Encoders with Memory

```python
# From moco/builder.py:

class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        super(MoCo, self).__init__()

        # Two identical encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        # Initialize momentum encoder with same weights as base encoder
        for param_b, param_m in zip(
            self.base_encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_m.data.copy_(param_b.data)  # Same initial weights
            param_m.requires_grad = False     # DON'T update this one!
```

### How MoCo Works:

**Step 1: Create Two Views of Same Image**

```python
# Take an image and apply two different augmentations (transformations)
# Augmentations: crop, rotate, color jitter, etc.

img = load_image("dog.jpg")  # Original image

# Augmentation 1: Random crop + color distortion
view1 = augment(img)  # Slightly modified version of image

# Augmentation 2: Different random crop + color distortion
view2 = augment(img)  # Different modification of same image

# Key: Same image, different views!
```

**Step 2: Encode Both Views**

```python
# Q: Queries (from base encoder, which DOES learn)
q1 = self.predictor(self.base_encoder(view1))
q2 = self.predictor(self.base_encoder(view2))

# K: Keys (from momentum encoder, which updates slowly)
with torch.no_grad():  # Don't calculate gradients
    k1 = self.momentum_encoder(view1)
    k2 = self.momentum_encoder(view2)
```

**Step 3: Contrastive Loss**

Compare features using a **contrastive loss**:

```python
def contrastive_loss(self, q, k):
    # Normalize (make unit vectors)
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    # Compute similarity matrix (q vs all k)
    # logits[i, j] = how similar query_i is to key_j
    logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
    # Divided by temperature T (controls sharpness)

    # Loss: queries should match their corresponding keys,
    # but be different from other keys (like cross-entropy)
    loss = CrossEntropyLoss()(logits, labels)

    return loss

# Total loss:
# Make sure q1 matches k2 (cross-images)
# Make sure q2 matches k1 (cross-images)
loss = contrastive_loss(q1, k2) + contrastive_loss(q2, k1)
```

**Result**: The network learns to recognize content despite view differences!

### Why Two Encoders?

```
Base encoder:     ← Gets gradient updates (learns)
Momentum encoder: ← Updated slowly with: param = param * m + base_param * (1-m)
                    where m ≈ 0.999 (99.9% old, 0.1% new)
```

**Why?**
- Momentum encoder provides "slowly moving targets"
- More stable than targets that change every iteration
- Encoders stay different but correlated (important for contrastive learning)

### Code from Your Project:

```python
# From moco/builder.py

@torch.no_grad()
def _update_momentum_encoder(self, m):
    """Momentum update of the momentum encoder"""
    for param_b, param_m in zip(
        self.base_encoder.parameters(),
        self.momentum_encoder.parameters()
    ):
        # Exponential moving average
        param_m.data = param_m.data * m + param_b.data * (1. - m)
        #            = 0.999 × old + 0.001 × new (if m=0.999)

def forward(self, x1, x2, m):
    # x1, x2: Two views of image
    # m: momentum coefficient

    # Compute features
    q1 = self.predictor(self.base_encoder(x1))
    q2 = self.predictor(self.base_encoder(x2))

    with torch.no_grad():  # No gradients for momentum encoder
        self._update_momentum_encoder(m)

        # Compute momentum features
        k1 = self.momentum_encoder(x1)
        k2 = self.momentum_encoder(x2)

    # Contrastive loss
    return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
```

---

## Temperature (T) Parameter

The temperature controls how "sharp" the loss is:

```python
logits = torch.einsum('nc,mc->nm', [q, k]) / T

# High T (0.07): Softer, more gradual gradients
# Low T (1.0): Sharper, more peaky
```

Lower temperature makes the model more confident about differences.

---

# PART 5: YOUR REIQA CODEBASE

Now let's understand how everything connects in your actual code!

## What is ReIQA?

**ReIQA** = **Re**ference-Free **I**mage **Q**uality **A**ssessment

It learns to understand image quality **without** reference images, using contrastive learning!

### Project Structure

```
ReIQA/
├── networks/              # Neural network architectures
│   ├── resnet.py         # ResNet implementation (we studied this!)
│   ├── resnest.py        # ResNest (improved ResNet)
│   └── build_backbone.py # Creates model instances
│
├── moco/                  # Momentum Contrast implementation
│   ├── builder.py        # MoCo class definition
│   ├── distortion_augmentations.py  # Image augmentations
│   ├── losses.py         # Loss functions
│   └── optimizer.py      # Special optimizers
│
├── datasets/             # Data loading
│   ├── dataset.py        # Creates batches of images
│   └── iqa_distortions.py # Image quality distortions
│
├── learning/             # Training logic
│   ├── base_trainer.py   # Base trainer class
│   ├── contrast_trainer.py # Contrastive training specific
│   └── linear_trainer.py # Linear evaluation trainer
│
├── memory/               # Memory bank (for contrastive learning)
│   ├── mem_bank.py       # Memory bank implementation
│   └── mem_moco.py       # MoCo memory variant
│
├── options/              # Command-line arguments
│   ├── train_options.py  # Training arguments
│   └── test_options.py   # Testing arguments
│
└── main_contrast.py      # Main training script
```

---

## Main Training Script (`main_contrast.py`)

Let's trace through the execution:

```python
# main_contrast.py

def main():
    # 1. Parse command-line arguments
    args = TrainOptions().parse()
    # Contains: learning_rate, batch_size, epochs, etc.

    # 2. Setup distributed training (multi-GPU)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    # 3. Launch training on all GPUs
    if args.multiprocessing_distributed:
        mp.spawn(
            main_worker,           # Function to run on each GPU
            nprocs=ngpus_per_node, # Number of GPUs
            args=(ngpus_per_node, args)
        )

def main_worker(gpu, ngpus_per_node, args):
    # This function runs on each GPU separately

    # 1. Initialize trainer
    trainer = ContrastTrainer(args)
    trainer.init_ddp_environment(gpu, ngpus_per_node)
    # DDP = Distributed Data Parallel (sync gradients between GPUs)

    # 2. Build model
    model, model_ema = build_model(args)
    # Two models: one to train, one momentum version

    # 3. Build dataset and dataloader
    train_dataset, train_loader, train_sampler = build_contrast_loader(args, ngpus_per_node)
    # Loads images, applies augmentations, creates batches

    # 4. Build criterion and optimizer
    criterion = nn.CrossEntropyLoss()  # Contrastive loss

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(...)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(...)
    elif args.optimizer == "LARS":
        optimizer = moco.optimizer.LARS(...)  # Large-batch optimizer

    # 5. Wrap model for DDP
    model, model_ema, optimizer = trainer.wrap_up(model, model_ema, optimizer)

    # 6. Resume from checkpoint if exists
    start_epoch = trainer.resume_model(model, model_ema, contrast, optimizer)

    # 7. Initialize tensorboard for monitoring
    trainer.init_tensorboard_logger()

    # 8. Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # For reproducibility in DDP
        trainer.adjust_learning_rate(optimizer, epoch)

        # Train one epoch
        outs = trainer.train(
            epoch, train_loader, model, model_ema,
            contrast, criterion, optimizer
        )

        # Log to tensorboard
        trainer.logging(epoch, outs, optimizer.param_groups[0]['lr'])

        # Save checkpoint
        trainer.save(model, model_ema, contrast, optimizer, epoch)
```

---

## Dataset Loading (`datasets/dataset.py`)

### Image Augmentations (Creating Two Views)

```python
# For each image, create two augmented views for contrastive learning

class ImageFolderInstance(datasets.ImageFolder):
    """Custom dataset that returns image index + two augmented views"""

    def __init__(self, root, transform=None, two_crop=False, jigsaw_transform=None):
        super(ImageFolderInstance, self).__init__(root, transform)
        self.two_crop = two_crop  # Create two separate augmented views?
        self.jigsaw_transform = jigsaw_transform  # Extra jigsaw augmentation

    def __getitem__(self, index):
        # Load actual image file
        path, target = self.imgs[index]
        image = self.loader(path)

        # Apply augmentation(s)
        if self.transform is not None:
            img = self.transform(image)  # First augmented view

            if self.two_crop:
                img2 = self.transform(image)  # DIFFERENT augmentation of same image!
                img = torch.cat([img, img2], dim=0)  # Concatenate: (6, 224, 224)
                #                                      instead of (3, 224, 224)

        # Optional: Jigsaw puzzle augmentation
        if self.use_jigsaw:
            jigsaw_image = self.jigsaw_transform(image)
            return img, index, jigsaw_image
        else:
            return img, index
```

### Why Two Views?

```
Contrastive learning NEEDS two versions of the same image:
- Same image + Augmentation A = View 1
- Same image + Augmentation B = View 2

Network learns: "These should produce similar features"
```

---

## Augmentations (`moco/distortion_augmentations.py`)

```python
# Image quality assessment needs special augmentations!
# We apply distortions like:

# 1. Gaussian blur
# 2. JPEG compression
# 3. Noise
# 4. Brightness/contrast changes

# Example augmentation pipeline:
augment_pipeline = transforms.Compose([
    transforms.RandomResizedCrop(224),      # Random crop
    transforms.RandomHorizontalFlip(p=0.5),  # Random flip
    transforms.ColorJitter(                  # Random color changes
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),
    transforms.GaussianBlur(kernel_size=3),  # Blur
    transforms.RandomApply([                 # Sometimes apply
        transforms.Compose([
            transforms.GaussianNoise(),       # Add noise
        ])
    ], p=0.1),
    transforms.ToTensor(),                   # Convert to tensor
    transforms.Normalize(                    # Normalize values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

---

## Training Loop (`learning/contrast_trainer.py`)

```python
class ContrastTrainer(BaseTrainer):
    def train(self, epoch, train_loader, model, model_ema, contrast, criterion, optimizer):
        """One training epoch"""

        batch_time = AverageMeter()
        data_time = AverageMeter()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        model.train()  # Set to training mode (enables dropout, BN updates)

        for i, (input, index, *other) in enumerate(train_loader):
            # input shape: (B*2, 3, 224, 224) if two_crop=True
            # index shape: (B,)

            input = input.cuda(non_blocking=True)  # Move to GPU
            index = index.cuda(non_blocking=True)

            # Get two views
            input1 = input[:len(input)//2]  # First half
            input2 = input[len(input)//2:]  # Second half

            # Forward pass
            # model returns: (query_feature, key_feature) for contrastive learning
            out1, target1 = model(input1, input2)
            out2, target2 = model(input2, input1)

            # Compute contrastive loss
            loss = criterion(out1, target1) + criterion(out2, target2)

            # Backward pass
            optimizer.zero_grad()   # Clear old gradients
            loss.backward()         # Calculate gradients
            optimizer.step()        # Update weights

            # Update momentum encoder
            model.module.update_momentum_encoder()
            # (model.module because of DDP wrapper)

            # Update memory bank with new features
            contrast.update(index, out1.detach(), out2.detach())

            # Logging
            loss_meter.update(loss.item(), input1.size(0))
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")

        return loss_meter.val
```

---

## Memory Bank

For large-scale contrastive learning, we can't compare with all images in the dataset. We use a **memory bank**:

```python
# From memory/mem_bank.py

class MemBank(nn.Module):
    """Stores features of all images for contrastive learning"""

    def __init__(self, dim, num_samples):
        super(MemBank, self).__init__()

        # Large tensor storing features for all images
        self.register_buffer(
            'memory',
            torch.randn(num_samples, dim).cuda()
        )
        # shape: (number_of_images, feature_dimension)
        # Example: (1_000_000, 128) for 1M images with 128-dim features

    def update(self, idx, features):
        """Update memory with new features"""
        self.memory[idx] = features  # Store new features
```

**Why memory bank?**
```
Without memory:
- Batch: 256 images
- Compare each with 256 others = 256² = 65,536 comparisons ✓ Fast

With memory:
- Batch: 256 images
- Compare each with 1,000,000 in memory = 256M comparisons ✗ Very slow!

Memory bank solution:
- Keep memory from previous iterations
- Compare batch with old memory
- Update memory as you train
- Still large-scale but manageable
```

---

# PART 6: COMPLETE WORKFLOW

Now let's see the complete picture of how everything works together:

## Training Workflow

### Step 1: Data Loading & Augmentation

```
Raw images
    ↓
Apply random augmentation → Image view 1 (224×224×3)
Apply random augmentation → Image view 2 (224×224×3)
    ↓
Batch: (32, 6, 224, 224)  ← 32 images × 2 views × 3 channels
```

### Step 2: Forward Pass Through Network

```
Image view 1 (B, 3, 224, 224)
    ↓
ResNet backbone (extracts features)
    ↓
Output: (B, 512 or 2048)  ← High-level features
    ↓
MLP projection head
    ↓
Output: (B, 128)  ← Contrastive features (same size for all images)

Same for Image view 2
```

### Step 3: Contrastive Loss

```
Features from view 1: q1 = (B, 128)
Features from view 2: q2 = (B, 128)
Memory bank features: K = (1_000_000, 128)

Compute similarity: logits = q1 @ K.T = (B, 1_000_000)
                              128  128×1_000_000

Cross-entropy loss between logits and target labels
Target: q1 should match view2 of same image, not other images

Loss = CrossEntropyLoss(logits, labels)
```

### Step 4: Backward Pass & Update

```
Loss
    ↓
Calculate gradients (backpropagation)
    ↓
Update base encoder weights
    ↓
Slowly update momentum encoder: param_m = 0.999 * param_m + 0.001 * param_b
```

### Step 5: Epoch Loop

```
For each epoch (e.g., 200 epochs):
    For each batch in dataset:
        1. Load image batch
        2. Create two augmented views
        3. Forward pass (get features)
        4. Compute contrastive loss
        5. Backward pass
        6. Update weights
        7. Update memory bank

    Save checkpoint
    Log metrics
```

---

## Complete Code Example: From Image to Prediction

Let's trace a single image through the entire process:

```python
# 1. LOAD IMAGE
image_path = "dog.jpg"
image = Image.open(image_path)  # PIL Image: (1200, 800)

# 2. AUGMENT (two views)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

view1 = transform(image)  # Tensor: (3, 224, 224)
view2 = transform(image)  # Tensor: (3, 224, 224) - DIFFERENT due to randomness!

# 3. CREATE BATCH
view1 = view1.unsqueeze(0)  # (1, 3, 224, 224)
view2 = view2.unsqueeze(0)  # (1, 3, 224, 224)
batch_view1 = view1.cuda()
batch_view2 = view2.cuda()

# 4. FORWARD PASS THROUGH ResNet
model = resnet50(pretrained=True)
model = model.cuda()

# From ResNet forward method:
x = batch_view1  # (1, 3, 224, 224)
x = model.conv1(x)           # (1, 64, 112, 112)
x = model.bn1(x)
x = model.relu(x)
x = model.maxpool(x)         # (1, 64, 56, 56)

x = model.layer1(x)          # (1, 64, 56, 56)
x = model.layer2(x)          # (1, 128, 28, 28)
x = model.layer3(x)          # (1, 256, 14, 14)
x = model.layer4(x)          # (1, 2048, 7, 7) for ResNet50

x = model.avgpool(x)         # (1, 2048, 1, 1)
x = torch.flatten(x, 1)      # (1, 2048)

# 5. PROJECTION HEAD
projection_head = nn.Linear(2048, 128)
features_view1 = projection_head(x)  # (1, 128)

# Repeat for view2...
features_view2 = projection_head(model(batch_view2))  # (1, 128)

# 6. NORMALIZE FEATURES
q1 = F.normalize(features_view1, dim=1)  # (1, 128) with norm=1
q2 = F.normalize(features_view2, dim=1)  # (1, 128) with norm=1

# 7. COMPUTE SIMILARITY WITH MEMORY
memory = torch.randn(1_000_000, 128).cuda()  # All images from dataset
memory = F.normalize(memory, dim=1)

similarity = torch.matmul(q1, memory.t())  # (1, 1_000_000)
# similarity[0, j] = how similar view1 is to image j in memory

# 8. COMPUTE LOSS
temperature = 0.07
logits = similarity / temperature  # (1, 1_000_000)

# Target: the feature from view2 of same image
target_label = 0  # Assuming memory[0] = features of view2

loss = nn.CrossEntropyLoss()(logits, torch.tensor([target_label]))

# 9. BACKWARD PASS
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Key Insights & Takeaways

### Why This Architecture?

1. **ResNet backbone**: Proven deep architecture that works well
2. **MoCo**: Efficient contrastive learning with momentum encoder
3. **Two views**: Model learns that augmented versions of same image are similar
4. **Memory bank**: Large-scale learning without comparing all image pairs
5. **Distributed training**: Use multiple GPUs to handle large batches

### When to Use Each Component

| Component | Purpose | When to Use |
|-----------|---------|------------|
| ResNet50 | Feature extraction | Default, balanced |
| ResNet101 | Feature extraction | Need more capacity |
| BasicBlock | Simple residual block | ResNet18/34 |
| Bottleneck | Efficient residual block | ResNet50+ |
| MoCo | Contrastive learning | Learning without labels |
| Memory bank | Large-scale similarity | Many images |
| DDP | Multi-GPU training | Multiple GPUs available |
| Temperature | Loss sharpness | Usually 0.07 for MoCo |

### Common Hyperparameters

```python
# ResNet
- width: 1.0 (normal), 0.5 (smaller)
- in_channel: 3 (RGB)

# Training
- batch_size: 256-1024 (larger batches = more stable)
- learning_rate: 0.03-0.3 (depends on batch size)
- momentum: 0.9 (for SGD optimizer)
- weight_decay: 0.0001-0.0004 (regularization)
- epochs: 100-200

# MoCo
- T (temperature): 0.07 (lower = sharper, more discriminative)
- mom_coefficient (m): 0.999 (0.9999 for very large batches)
- mlp_dim: 4096 (hidden dimension in projection head)
- feature_dim: 128-256 (final feature dimension)

# Augmentations
- crop_size: 224
- gaussian_blur: p=0.5 (probability)
- color_jitter: 0.4 (strength)
```

---

## Troubleshooting Guide

### Problem: Training loss doesn't decrease

**Possible causes:**
- Learning rate too high (loss explodes)
- Learning rate too low (loss barely changes)
- **Solution**: Start with 0.03, watch first 100 steps

### Problem: Memory runs out

**Possible causes:**
- Batch size too large
- Image resolution too high
- **Solution**: Reduce batch_size or image_size

### Problem: Multi-GPU not working

**Possible causes:**
- Not using DistributedDataParallel (DDP)
- Gradient not synchronized between GPUs
- **Solution**: Check ContrastTrainer wraps model in DDP

### Problem: Features not discriminative

**Possible causes:**
- Temperature T too high (loss too soft)
- Too few augmentations (views too similar)
- **Solution**: Lower T, add more augmentations

---

## Next Steps: What to Explore

1. **Modify augmentations**: Try different image distortions
2. **Change architecture**: Try ResNet101, ResNeSt, or Vision Transformers
3. **Tune hyperparameters**: Play with learning rate, batch size, temperature
4. **Add monitoring**: Track feature statistics, similarity distribution
5. **Fine-tune on downstream**: After pre-training, use features for quality assessment task

---

## Quick Reference: Common PyTorch Patterns

```python
# Create model
model = resnet50(pretrained=True)

# Move to GPU
model = model.cuda()

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)

# Training loop
for epoch in range(epochs):
    for images, labels in dataloader:
        # Forward
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save checkpoint
torch.save(model.state_dict(), 'model.pth')

# Load checkpoint
model.load_state_dict(torch.load('model.pth'))

# Inference
model.eval()
with torch.no_grad():
    predictions = model(test_images)
```

---

## Glossary of Terms

- **Tensor**: Multi-dimensional array
- **Epoch**: One pass through entire dataset
- **Batch**: Subset of data processed together
- **Gradient**: Direction to update weights
- **Backpropagation**: Algorithm to compute gradients
- **Loss**: Measure of prediction error
- **Optimizer**: Algorithm to update weights based on gradients
- **Learning rate**: Step size for weight updates
- **Convolution**: Sliding filter operation on images
- **Residual block**: Layer that adds input to output
- **Skip connection**: Shortcut connection in residual block
- **Normalization**: Scaling values to have mean=0, std=1
- **Activation function**: Non-linearity applied after layer
- **Augmentation**: Random transformation of data
- **Contrastive learning**: Learning by comparing similar/different pairs
- **Momentum encoder**: Slowly updated copy of main encoder
- **Memory bank**: Stored features for large-scale learning
- **Distributed training**: Training across multiple GPUs
- **Cross-entropy loss**: Classification loss function
- **Temperature**: Hyperparameter controlling prediction sharpness

---

## Final Thoughts

Congratulations! You now understand:

✓ **PyTorch basics** - Tensors, gradients, optimization
✓ **Neural networks** - Layers, convolutions, activations
✓ **ResNet architecture** - Skip connections, deep networks
✓ **Contrastive learning** - Learning from unlabeled data
✓ **MoCo** - Practical implementation with momentum encoder
✓ **Your ReIQA codebase** - How all pieces fit together
✓ **Training pipeline** - From images to learned features

The best way to learn further is:
1. Run the code with different hyperparameters
2. Print intermediate tensor shapes to understand flow
3. Visualize learned features and similarities
4. Read research papers (start with ResNet and MoCo papers)
5. Experiment with modifications!

Happy learning!
