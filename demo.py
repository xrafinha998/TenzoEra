"""
TensorEra — Complete Feature Demo
===================================
Showcases all major capabilities: tensors, autograd, nn layers,
optimizers, data pipeline, and training loop.
"""

import sys
sys.path.insert(0, "..")

import tensorera as te
import tensorera.nn as nn
import tensorera.optim as optim

te.manual_seed(42)
print()

# ═══════════════════════════════════════════════════════════
# 1. TENSORS & OPERATIONS
# ═══════════════════════════════════════════════════════════
print("━" * 50)
print("  1. TENSORS & OPERATIONS")
print("━" * 50)

x = te.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(f"Shape     : {x.shape}")
print(f"Sum       : {x.sum().item():.1f}")
print(f"Mean      : {x.mean().item():.2f}")
print(f"Transpose :")
print(x.T.data)

a = te.randn(3, 4)
b = te.randn(4, 2)
c = a @ b  # (3, 4) @ (4, 2) = (3, 2)
print(f"\nMatrix multiply: {a.shape} @ {b.shape} = {c.shape}")

# ═══════════════════════════════════════════════════════════
# 2. AUTOGRAD
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  2. AUTOGRAD")
print("━" * 50)

w = te.tensor([[2.0, -1.0], [3.0, 0.5]], requires_grad=True)
x = te.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

y = (w * x).sum()
y.backward()

print(f"y = (w * x).sum() = {y.item()}")
print(f"dy/dw = \n{w.grad.data}")
print(f"dy/dx = \n{x.grad.data}")

# ═══════════════════════════════════════════════════════════
# 3. NEURAL NETWORK LAYERS
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  3. NEURAL NETWORK")
print("━" * 50)

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

print(f"Parameters: {model.count_parameters():,}")
dummy_input = te.randn(16, 784)
output = model(dummy_input)
print(f"Input  shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")

# ═══════════════════════════════════════════════════════════
# 4. ALL ACTIVATIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  4. ACTIVATION FUNCTIONS")
print("━" * 50)

z = te.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
activations = {
    "ReLU"     : te.relu(z),
    "LeakyReLU": te.leaky_relu(z),
    "Sigmoid"  : te.sigmoid(z),
    "Tanh"     : te.tanh(z),
    "GELU"     : te.gelu(z),
    "SiLU"     : te.silu(z),
    "Mish"     : te.mish(z),
    "Softmax"  : te.softmax(z),
}
for name, out in activations.items():
    print(f"  {name:12s}: {[f'{v:.3f}' for v in out.tolist()]}")

# ═══════════════════════════════════════════════════════════
# 5. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  5. LOSS FUNCTIONS")
print("━" * 50)

pred   = te.randn(8, 10, requires_grad=True)
target = te.randint(0, 10, size=(8,))
reg_target = te.randn(8)
reg_pred   = te.randn(8, requires_grad=True)

losses = {
    "CrossEntropy" : te.cross_entropy_loss(pred, target),
    "MSE"          : te.mse_loss(reg_pred, reg_target),
    "MAE"          : te.mae_loss(reg_pred, reg_target),
    "Huber"        : te.huber_loss(reg_pred, reg_target),
}
for name, loss in losses.items():
    print(f"  {name:15s}: {loss.item():.4f}")

# ═══════════════════════════════════════════════════════════
# 6. OPTIMIZERS
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  6. OPTIMIZERS")
print("━" * 50)

optimizer_names = ["SGD", "Adam", "AdamW", "RMSprop", "Adagrad"]
print(f"  Available: {optimizer_names}")

# ═══════════════════════════════════════════════════════════
# 7. DATA PIPELINE
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  7. DATA PIPELINE")
print("━" * 50)

N = 500
X = te.randn(N, 20)
y = te.randint(0, 3, size=(N,))
dataset = te.TensorDataset(X, y)
train_ds, val_ds = te.random_split(dataset, [400, 100], seed=0)

train_loader = te.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = te.DataLoader(val_ds,   batch_size=32)

print(f"  Dataset size : {len(dataset)}")
print(f"  Train / Val  : {len(train_ds)} / {len(val_ds)}")
print(f"  Train batches: {len(train_loader)}")

batch_X, batch_y = next(iter(train_loader))
print(f"  Batch X shape: {batch_X.shape}")
print(f"  Batch y shape: {batch_y.shape}")

# ═══════════════════════════════════════════════════════════
# 8. FULL TRAINING LOOP
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  8. TRAINING LOOP")
print("━" * 50)

class SimpleNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


clf   = SimpleNet(20, 64, 3)
opt   = optim.Adam(clf.parameters(), lr=5e-3)
sched = optim.StepLR(opt, step_size=5, gamma=0.5)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(15):
    clf.train()
    total_loss = 0
    for xb, yb in train_loader:
        opt.zero_grad()
        out  = clf(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        te.clip_grad_norm_(clf.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item()
    sched.step()

    clf.eval()
    correct = total = 0
    for xb, yb in val_loader:
        preds = te.argmax(clf(xb), dim=1)
        correct += int((preds == yb).data.sum())
        total += len(yb)

    if (epoch + 1) % 5 == 0:
        lr = opt.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | "
              f"Val Acc: {correct/total*100:.1f}% | LR: {lr:.5f}")

# ═══════════════════════════════════════════════════════════
# 9. DEVICE INFO
# ═══════════════════════════════════════════════════════════
print("\n" + "━" * 50)
print("  9. DEVICE SUPPORT")
print("━" * 50)
print(f"  Best device  : {te.get_device()}")
print(f"  CUDA available: {te.cuda_is_available()}")

t = te.randn(3, 3)
print(f"  CPU tensor device: {t.device}")

print()
print("✅  TensorEra demo complete!")
