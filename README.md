# 🔥 TensorEra

> **A productive PyTorch-like deep learning framework that runs on any device.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/backend-NumPy-orange.svg)](https://numpy.org/)

TensorEra is a full-featured deep learning framework inspired by PyTorch, built from the ground up with a pure NumPy backend. It provides a familiar API for building, training, and deploying neural networks — **without complex GPU driver setup**. Optional CUDA acceleration via CuPy.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧮 **Tensors** | N-dimensional arrays with broadcasting, slicing, and in-place ops |
| 🔁 **Autograd** | Reverse-mode automatic differentiation (backpropagation) |
| 🧱 **nn Modules** | Linear, Conv2d, LSTM, GRU, Attention, BatchNorm, LayerNorm, ... |
| ⚡ **Optimizers** | SGD, Adam, AdamW, RMSprop, Adagrad, Adadelta, LBFGS |
| 📅 **Schedulers** | StepLR, CosineAnnealingLR, OneCycleLR, WarmupScheduler, ... |
| 📦 **Data** | Dataset, DataLoader, transforms, samplers — full pipeline |
| 📱 **Devices** | CPU always works; CUDA/MPS/ROCm via optional backends |
| 💾 **Save/Load** | Checkpoint models with `te.save` / `te.load` |

---

## 📦 Installation

### Option 1 — Minimal (CPU only)
```bash
pip install numpy
pip install -e .
```

### Option 2 — With CUDA support (NVIDIA GPU)
```bash
pip install cupy-cuda12x   # Match your CUDA version
pip install -e .
```

### Option 3 — Full install
```bash
pip install -e ".[full]"
```

### Option 4 — From PyPI (when published)
```bash
pip install tensorera
```

---

## 🚀 Quick Start

```python
import tensorera as te
import tensorera.nn as nn
import tensorera.optim as optim

# ── 1. Create Tensors ───────────────────────────────────────────────
x = te.randn(3, 4)
y = te.ones(4, 5)
z = x @ y
print(z)  # tensor([[...]])

# ── 2. Autograd ─────────────────────────────────────────────────────
a = te.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = te.tensor([[2.0], [3.0]], requires_grad=True)
c = (a @ b).sum()
c.backward()
print(a.grad)  # dc/da

# ── 3. Build a model ────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)
print(f"Parameters: {model.count_parameters():,}")

# ── 4. Train ────────────────────────────────────────────────────────
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

X = te.randn(32, 784)
y = te.randint(0, 10, size=(32,))

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X)
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
```

---

## 📚 API Reference

### Tensor creation
```python
te.tensor([1, 2, 3])          # From data
te.zeros(3, 4)                 # All zeros
te.ones(3, 4)                  # All ones
te.randn(3, 4)                 # Standard normal
te.rand(3, 4)                  # Uniform [0, 1)
te.randint(0, 10, (3, 4))      # Random integers
te.eye(4)                      # Identity matrix
te.arange(0, 10, 2)            # Like np.arange
te.linspace(0, 1, 100)         # Like np.linspace
```

### Tensor operations
```python
# Arithmetic (full broadcasting)
z = x + y
z = x * y
z = x @ y           # Matrix multiply
z = x ** 2          # Element-wise power

# Shape
x.reshape(4, -1)
x.squeeze(0)
x.unsqueeze(0)
te.cat([a, b], dim=0)
te.stack([a, b], dim=0)
te.transpose(x, 0, 1)

# Reductions
x.sum()
x.mean(axis=0)
x.max()
te.argmax(x, dim=1)
```

### Neural network layers
```python
nn.Linear(in_features, out_features)
nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
nn.BatchNorm1d(num_features)
nn.LayerNorm(normalized_shape)
nn.Embedding(vocab_size, embed_dim)
nn.MultiheadAttention(embed_dim, num_heads)
nn.LSTMCell(input_size, hidden_size)
nn.GRUCell(input_size, hidden_size)
nn.Dropout(p=0.5)
nn.Sequential(layer1, layer2, ...)
```

### Activation functions
```python
nn.ReLU()        # Rectified Linear
nn.LeakyReLU()   # Leaky ReLU
nn.GELU()        # Gaussian Error LU
nn.SiLU()        # Swish
nn.Sigmoid()     # Logistic
nn.Tanh()        # Hyperbolic tangent
nn.Softmax(dim)  # Softmax
nn.Mish()        # Mish activation
```

### Loss functions
```python
nn.MSELoss()           # Mean Squared Error
nn.CrossEntropyLoss()  # For classification
nn.BCELoss()           # Binary Cross Entropy
nn.HuberLoss()         # Robust regression loss
nn.MAELoss()           # Mean Absolute Error
```

### Optimizers
```python
optim.SGD(params, lr=0.01, momentum=0.9)
optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
optim.AdamW(params, lr=1e-3, weight_decay=0.01)
optim.RMSprop(params, lr=1e-2)
optim.Adagrad(params, lr=1e-2)
```

### Learning rate schedulers
```python
optim.StepLR(optimizer, step_size=10, gamma=0.1)
optim.CosineAnnealingLR(optimizer, T_max=100)
optim.ReduceLROnPlateau(optimizer, patience=5)
optim.OneCycleLR(optimizer, max_lr=1e-2, total_steps=1000)
optim.WarmupScheduler(optimizer, warmup_steps=100, total_steps=1000)
```

### Data pipeline
```python
dataset = te.TensorDataset(X, y)
train_set, val_set = te.random_split(dataset, [800, 200])
loader = te.DataLoader(train_set, batch_size=32, shuffle=True)

for X_batch, y_batch in loader:
    ...

# Transforms
transform = te.Compose([
    te.ToTensor(),
    te.Normalize(mean=0.5, std=0.5),
    te.RandomHorizontalFlip(p=0.5),
])
```

### Device management
```python
device = te.get_device()       # Auto-select best device
print(te.cuda_is_available())  # True if CUDA found

x = te.randn(3, 4)
x = x.cuda()                  # Move to GPU
x = x.cpu()                   # Back to CPU
model.to("cuda")               # Move entire model
```

### Saving & loading
```python
# Save model checkpoint
te.save(model.state_dict(), "checkpoint.npz")

# Load checkpoint
state = te.load("checkpoint.npz")
model.load_state_dict(state)
```

---

## 🔬 Complete Training Example — MNIST-like Classifier

```python
import tensorera as te
import tensorera.nn as nn
import tensorera.optim as optim

te.manual_seed(42)

# Generate synthetic data
N, D, C = 1000, 784, 10
X = te.randn(N, D)
y = te.randint(0, C, size=(N,))

dataset = te.TensorDataset(X, y)
train_set, val_set = te.random_split(dataset, [800, 200], seed=42)
train_loader = te.DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = te.DataLoader(val_set, batch_size=64)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


model = MLP()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.CosineAnnealingLR(optimizer, T_max=20)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    train_loss = 0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        out = model(X_b)
        loss = loss_fn(out, y_b)
        loss.backward()
        te.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()

    # Validation
    model.eval()
    correct = total = 0
    for X_b, y_b in val_loader:
        out = model(X_b)
        preds = te.argmax(out, dim=1)
        correct += (preds == y_b).data.sum()
        total += len(y_b)

    print(f"Epoch {epoch+1:2d} | Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Acc: {correct/total*100:.1f}%")
```

---

## 🛠 Architecture

```
tensorera/
├── tensor.py       Core Tensor class + autograd engine
├── functional.py   Stateless functions (activations, losses, etc.)
├── nn.py           Neural network modules (Layer, Model, Loss)
├── optim.py        Optimizers + LR schedulers
├── ops.py          Tensor creation + math + utilities
└── data.py         Dataset, DataLoader, transforms
```

---

## 🤝 Contributing

TensorEra is open source and welcomes contributions!

```bash
git clone https://github.com/tensorera/tensorera
cd tensorera
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 📄 License

MIT License © 2025 TensorEra Contributors
