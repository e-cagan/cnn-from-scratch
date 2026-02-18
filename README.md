# CNN from Scratch: A Deep Learning Implementation Journey

A production-grade Convolutional Neural Network implementation using **pure NumPy**, achieving **98.33% test accuracy** on MNIST without any deep learning frameworks.

## 🎯 Project Overview

This project implements a complete CNN training pipeline from first principles, including:
- Custom layer implementations (Conv, MaxPool, FC, ReLU, Flatten, Softmax)
- Vectorized operations using im2col and stride tricks
- Multiple optimizers (SGD, SGD+Momentum, Adam)
- Gradient verification system
- Full training and evaluation pipeline

**Final Results:**
- **Test Accuracy: 98.33%**
- **Validation Accuracy: 98.15%** (best epoch)
- **Training Loss: 0.0456** (final epoch)

---

## 💡 Motivation: No Vibe-Coding

This project follows a **"zero vibe-coding"** philosophy:
- Every line of code is understood and justified
- Mathematical foundations are explored before implementation
- Gradient checking verifies correctness at each layer
- Progressive optimization (naive → vectorized)

The goal: **understand CNNs at the lowest level**, not just use them.

---

## 🏗️ Architecture

```
Input (1, 28, 28)
    ↓
Conv2D(1→32, 5×5, padding=2) + ReLU + MaxPool(2×2)
    ↓
Conv2D(32→64, 5×5, padding=2) + ReLU + MaxPool(2×2)
    ↓
Flatten (3136)
    ↓
FC(3136→128) + ReLU
    ↓
FC(128→10) + Softmax
    ↓
Cross-Entropy Loss
```

**Total Parameters:** ~455K

---

## 🚀 Key Technical Achievements

### 1. Mathematical Correctness
Every layer passed numerical gradient checking with error < 1e-7:
```
FC Layer:   max_error=4.93e-11 → PASSED
Conv Layer: max_error=9.74e-12 → PASSED
```

### 2. Performance Optimization

**Naive Implementation → Vectorized**

| Operation | Naive | Vectorized | Speedup |
|-----------|-------|------------|---------|
| Conv Forward | 7 nested loops | im2col + matmul | ~15-20x |
| MaxPool Forward | 4 nested loops | as_strided + axis ops | ~10-15x |

**im2col Technique:**
```
Windows: (batch×out_H×out_W, in_channels×k×k)
Filters: (in_channels×k×k, out_channels)
Output:  (batch×out_H×out_W, out_channels)  ← Single matmul!
```

**Stride Tricks (MaxPool):**
```python
windows = as_strided(x, shape=(B,C,H',W',p,p), strides=(...))
output = windows.max(axis=(4,5))  # Vectorized max over all windows
```

### 3. Training Dynamics

**Loss Curve:**
```
Epoch  1: 1.2785 → Epoch 20: 0.0456 (97% reduction)
```

**Accuracy Progression:**
```
Epoch  1: 86.30% val
Epoch  5: 95.90% val
Epoch 10: 97.47% val
Epoch 14: 98.08% val (best)
Epoch 20: 97.62% val
Test:     98.33%
```

### 4. Model Generalization

**Confusion Matrix Analysis:**
- Diagonal dominance: Strong per-class performance
- Hardest confusion: 4→9 (11 errors), 9→4 (8 errors)
- Best performance: Class 1 (99.6% accuracy)
- Near-perfect: Classes 0, 6, 7 (>97% each)

---

## 📁 Project Structure

```
cnn-from-scratch/
├── layers/
│   ├── base_layer.py          # Abstract layer interface
│   ├── conv.py                # Naive convolution
│   ├── conv_vec.py            # Vectorized convolution (im2col)
│   ├── maxpool.py             # Naive max pooling
│   ├── maxpool_vec.py         # Vectorized pooling (as_strided)
│   ├── fc.py                  # Fully connected layer
│   ├── relu.py                # ReLU activation
│   ├── flatten.py             # Reshape layer
│   └── softmax.py             # Softmax + Cross-Entropy
├── optimizers/
│   ├── base_optimizer.py      # Abstract optimizer
│   ├── sgd.py                 # Vanilla SGD
│   ├── momentum.py            # SGD + Momentum
│   └── adam.py                # Adam optimizer
├── models/
│   ├── base_model.py          # Model interface
│   └── cnn.py                 # CNN architecture
├── utils/
│   ├── gradient_check.py      # Numerical gradient verification
│   ├── metrics.py             # Accuracy, confusion matrix
│   └── visualization.py       # Training curves, predictions
├── data/
│   └── load_mnist.py          # Data loading & preprocessing
├── tests/
│   ├── test_fc.py             # FC layer gradient check
│   └── test_conv.py           # Conv layer gradient check
├── train.py                    # Training loop
├── evaluate.py                 # Test set evaluation
├── inference.py                # Single image prediction
└── checkpoints/                # Saved models
```

---

## 🔬 Implementation Deep Dive

### Forward Pass: im2col Transformation

**Problem:** Nested loops make convolution O(n⁷) in Python.

**Solution:** Transform windows into columns, use BLAS-optimized matmul.

```python
# Extract all windows as rows
col = im2col(input, kernel_size, stride, padding)
# Shape: (batch×out_H×out_W, in_channels×k×k)

# Reshape filters as columns
W_col = weights.reshape(out_channels, -1).T
# Shape: (in_channels×k×k, out_channels)

# Single matrix multiplication replaces 7 loops
output = col @ W_col
# Shape: (batch×out_H×out_W, out_channels)
```

### Backward Pass: Chain Rule via Matmul

```python
# Gradient w.r.t. weights
dW = col.T @ dout_col  # Accumulate over all windows

# Gradient w.r.t. input
dX_col = dout_col @ W_col
dX = col2im(dX_col, input_shape, ...)  # Scatter back to image
```

### MaxPool: Memory-Efficient Window Views

```python
# Create 6D view without copying data
strides = (batch_stride, channel_stride, 
           stride×row_stride, stride×col_stride,
           row_stride, col_stride)
windows = as_strided(x, shape=(B,C,H',W',p,p), strides=strides)

# Vectorized max over last 2 dims
output = windows.max(axis=(4,5))
```

### Numerical Gradient Checking

Verify analytical gradients against finite differences:

```python
numerical = (f(θ + h) - f(θ - h)) / (2h)
analytical = backprop(θ)
relative_error = |analytical - numerical| / (|analytical| + |numerical|)

✅ Pass: error < 1e-7
⚠️  Acceptable: error < 1e-5
❌ Fail: error > 1e-3
```

---

## 📊 Training Results

### Loss & Accuracy Curves

```
Training Loss:
1.28 ███████████████████████████░░░░░░░░░░░░░ 0.05
     ↓ Smooth exponential decay

Validation Accuracy:
86.3% ░░░░░░░░░░░░░░░░░░░░░░░░░███████████████ 98.2%
      ↑ Steady improvement, no overfitting
```

### Convergence Analysis

- **Fast initial learning:** 86% → 95% in 5 epochs
- **Fine-tuning phase:** 95% → 98% over 15 epochs
- **No overfitting:** Train and val curves track closely
- **Stable plateau:** Best model at epoch 14, minimal variance after

### Per-Class Performance

| Digit | Accuracy | Common Errors |
|-------|----------|---------------|
| 0 | 98.9% | → 8 (2 cases) |
| 1 | 99.6% | → 2 (1 case) |
| 2 | 96.6% | → 9 (9 cases) |
| 3 | 99.3% | → 5 (6 cases) |
| 4 | 98.4% | → 9 (11 cases) |
| 5 | 98.0% | → 3 (3 cases) |
| 6 | 98.8% | → 0 (6 cases) |
| 7 | 99.4% | → 2 (8 cases) |
| 8 | 97.7% | → 5 (8 cases) |
| 9 | 96.4% | → 4 (8 cases) |

**Insight:** Digits with similar strokes (4↔9, 3↔5, 7↔2) show highest confusion.

---

## 🛠️ How to Run

### Prerequisites
```bash
python >= 3.8
numpy >= 1.21.0
matplotlib >= 3.4.0
torchvision (MNIST download only)
```

### Installation
```bash
git clone https://github.com/yourusername/cnn-from-scratch.git
cd cnn-from-scratch
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

### Training
```bash
python3 train.py
```

**Output:**
```
Epoch 1/20 | Loss: 1.2785 | Val Acc: 0.8630
Best model saved (val_acc: 0.8630)
...
Epoch 20/20 | Loss: 0.0456 | Val Acc: 0.9762
Test Accuracy: 0.9833
```

### Evaluation
```bash
python evaluate.py  # Confusion matrix & test metrics
python inference.py  # Single image prediction
```

### Gradient Checking
```bash
python tests/test_fc.py
python tests/test_conv.py
```

---

## 🔮 Future Improvements

### Features to Add
- [ ] Batch Normalization
- [ ] Dropout regularization
- [ ] Data augmentation (rotation, translation)
- [ ] Learning rate scheduling
- [ ] More architectures (ResNet blocks, deeper networks)

### Optimization
- [ ] Full im2col without any loops (even spatial)
- [ ] CUDA/GPU support via CuPy
- [ ] Mixed precision training
- [ ] Depthwise separable convolutions

### Experiments
- [ ] CIFAR-10/100 dataset
- [ ] Transfer learning experiments
- [ ] Pruning and quantization
- [ ] Adversarial robustness testing

---

## 📚 What I Learned

### Technical Skills
- **Deep understanding of backpropagation:** Not just chain rule, but how it flows through each layer type
- **Numerical stability:** Softmax overflow, gradient clipping, weight initialization
- **Vectorization techniques:** im2col, stride tricks, einsum operations
- **Debugging neural networks:** Gradient checking, shape tracking, loss curve analysis

### Engineering Practices
- **Test-driven development:** Gradient checks before full training
- **Progressive optimization:** Naive → verified → optimized
- **Memory management:** View vs copy, cache strategy
- **Modular design:** Abstract base classes, clean interfaces

### Mathematical Insights
- **Convolution as matmul:** Spatial operations → linear algebra
- **He initialization:** Why variance scaling matters for ReLU
- **Adam's bias correction:** Initial steps need special handling
- **Cross-entropy + softmax gradient:** Beautiful cancellation to `p - y`

---

## 🙏 Acknowledgments

**Educational Resources:**
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
- "Neural Networks and Deep Learning" by Michael Nielsen
- NumPy documentation and stride tricks guide

**Inspiration:**
- Andrej Karpathy's "micrograd" philosophy
- Yann LeCun's original LeNet architecture
- "Implementing CNNs from Scratch" (various blog posts)

---

## 📈 Results Summary

```
┌─────────────────────────────────────────┐
│  Final Model Performance                │
├─────────────────────────────────────────┤
│  Test Accuracy:        98.33%           │
│  Validation Accuracy:  98.15%  (best)   │
│  Training Loss:        0.0456  (final)  │
│  Total Parameters:     ~455K            │
│  Training Time:        ~20 epochs       │
│  Gradient Check:       ✅ All passed    │
└─────────────────────────────────────────┘
```

**Status:** ✅ Production-grade CNN implementation completed. No frameworks, pure understanding.

---

## 📝 License

MIT License - Feel free to use this for learning and educational purposes.

---

## 👤 Author

**Emin Çağan Apaydın** - Computer Engineering Student, Istanbul Okan University
- Focus: Computer Vision, Robotics, Deep Learning
- Project: CNN From Scratch - A deep dive into the inner workings of convolutional neural networks, built from the ground up with NumPy.

*"Understanding every line, no vibe-coding."*

---

**Star ⭐ this repo if you found it helpful!**