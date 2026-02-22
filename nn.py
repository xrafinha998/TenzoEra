"""
TensorEra - Neural Network Modules (te.nn)
PyTorch-style module system with full composability.
"""

import numpy as np
from typing import Optional, List, Dict, Iterator, Tuple, Union
from .tensor import Tensor
from . import functional as F


class Parameter(Tensor):
    """A Tensor that is considered a module parameter."""
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self._is_leaf = True


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = {}
        self._modules: Dict[str, Optional["Module"]] = {}
        self._buffers: Dict[str, Optional[Tensor]] = {}
        self.training: bool = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if "_parameters" in self.__dict__ and name in self._parameters:
            return self._parameters[name]
        if "_modules" in self.__dict__ and name in self._modules:
            return self._modules[name]
        raise AttributeError(f"Module has no attribute '{name}'")

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            if p is not None:
                yield p
        for m in self._modules.values():
            if m is not None:
                yield from m.parameters()

    def named_parameters(self, prefix="") -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            if p is not None:
                yield f"{prefix}.{name}" if prefix else name, p
        for mod_name, m in self._modules.items():
            if m is not None:
                pref = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from m.named_parameters(pref)

    def children(self) -> Iterator["Module"]:
        for m in self._modules.values():
            if m is not None:
                yield m

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            if m is not None:
                yield from m.modules()

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for m in self._modules.values():
            if m is not None:
                m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def to(self, device) -> "Module":
        for p in self.parameters():
            p.data = p.data  # CPU always in NumPy backend
        return self

    def state_dict(self) -> Dict:
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.data.copy()
        return sd

    def load_state_dict(self, state_dict: Dict, strict: bool = True):
        for name, p in self.named_parameters():
            if name in state_dict:
                p.data = state_dict[name].copy()
            elif strict:
                raise KeyError(f"Missing key: {name}")

    def __repr__(self):
        lines = [self.__class__.__name__ + "("]
        for name, m in self._modules.items():
            mod_str = repr(m)
            mod_str = "  " + mod_str.replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines) if self._modules else self.__class__.__name__ + "()"

    def count_parameters(self) -> int:
        return sum(p.size for p in self.parameters())


# ======================================================================
#  Core layers
# ======================================================================

class Linear(Module):
    """Fully connected linear layer: y = xW^T + b"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming uniform init
        k = np.sqrt(1 / in_features)
        self.weight = Parameter(np.random.uniform(-k, k, (out_features, in_features)))
        self.bias = Parameter(np.random.uniform(-k, k, (out_features,))) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.bias is not None})"


class Embedding(Module):
    """Lookup table for embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(np.random.normal(0, 1, (num_embeddings, embedding_dim)))
        if padding_idx is not None:
            self.weight.data[padding_idx] = 0

    def forward(self, indices: Tensor) -> Tensor:
        idx = indices.data.astype(int)
        return Tensor(self.weight.data[idx])

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


class Conv2d(Module):
    """2D Convolutional layer."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple],
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        k = 1 / (in_channels * kernel_size[0] * kernel_size[1])
        self.weight = Parameter(np.random.uniform(
            -np.sqrt(k), np.sqrt(k), (out_channels, in_channels, kernel_size[0], kernel_size[1])))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        # Simple 2D convolution using im2col
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        p = self.padding
        s = self.stride
        oH = (H + 2 * p - kH) // s + 1
        oW = (W + 2 * p - kW) // s + 1

        if p > 0:
            xp = np.pad(x.data, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            xp = x.data

        out = np.zeros((N, self.out_channels, oH, oW))
        for i in range(oH):
            for j in range(oW):
                patch = xp[:, :, i*s:i*s+kH, j*s:j*s+kW]  # N, C, kH, kW
                patch_flat = patch.reshape(N, -1)            # N, C*kH*kW
                w_flat = self.weight.data.reshape(self.out_channels, -1)  # OC, C*kH*kW
                out[:, :, i, j] = patch_flat @ w_flat.T     # N, OC

        if self.bias is not None:
            out += self.bias.data[None, :, None, None]
        return Tensor(out)

    def __repr__(self):
        return (f"Conv2d({self.in_channels}, {self.out_channels}, "
                f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})")


class ConvTranspose2d(Module):
    """Transposed 2D convolution (upsampling)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k = np.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = Parameter(np.random.uniform(-k, k, (in_channels, out_channels, *kernel_size)))
        self.bias = Parameter(np.zeros(out_channels)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        # Simple nearest-neighbor upsampling fallback
        N, C, H, W = x.data.shape
        s = self.stride
        out = np.repeat(np.repeat(x.data, s, axis=2), s, axis=3)
        return Tensor(out[:, :self.out_channels])


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        oH = (H - kH) // sH + 1
        oW = (W - kW) // sW + 1
        out = np.zeros((N, C, oH, oW))
        for i in range(oH):
            for j in range(oW):
                out[:, :, i, j] = x.data[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW].max(axis=(2, 3))
        return Tensor(out)


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride or self.kernel_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        kH, kW = self.kernel_size
        sH, sW = self.stride
        oH = (H - kH) // sH + 1
        oW = (W - kW) // sW + 1
        out = np.zeros((N, C, oH, oW))
        for i in range(oH):
            for j in range(oW):
                out[:, :, i, j] = x.data[:, :, i*sH:i*sH+kH, j*sW:j*sW+kW].mean(axis=(2, 3))
        return Tensor(out)


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        oH, oW = self.output_size
        sH, sW = H // oH, W // oW
        out = np.zeros((N, C, oH, oW))
        for i in range(oH):
            for j in range(oW):
                out[:, :, i, j] = x.data[:, :, i*sH:(i+1)*sH, j*sW:(j+1)*sW].mean(axis=(2, 3))
        return Tensor(out)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        shape = x.data.shape
        end = len(shape) if self.end_dim == -1 else self.end_dim + 1
        new_shape = shape[:self.start_dim] + (-1,) + shape[end:]
        return Tensor(x.data.reshape(new_shape))


# ======================================================================
#  Normalization layers
# ======================================================================

class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=0)
            var = x.data.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        return Tensor(x_norm * self.weight.data + self.bias.data)


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones(num_features))
        self.bias = Parameter(np.zeros(num_features))
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.data.mean(axis=(0, 2, 3))
            var = x.data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var
        x_norm = (x.data - mean[None, :, None, None]) / np.sqrt(var[None, :, None, None] + self.eps)
        return Tensor(x_norm * self.weight.data[None, :, None, None] + self.bias.data[None, :, None, None])


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = Parameter(np.ones(normalized_shape))
        self.bias = Parameter(np.zeros(normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape})"


class GroupNorm(Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = Parameter(np.ones(num_channels))
        self.bias = Parameter(np.zeros(num_channels))

    def forward(self, x: Tensor) -> Tensor:
        N, C, *spatial = x.data.shape
        G = self.num_groups
        x_r = x.data.reshape(N, G, -1)
        mean = x_r.mean(axis=-1, keepdims=True)
        var = x_r.var(axis=-1, keepdims=True)
        x_norm = (x_r - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, *spatial)
        return Tensor(x_norm * self.weight.data[None, :, *([None]*len(spatial))]
                      + self.bias.data[None, :, *([None]*len(spatial))])


# ======================================================================
#  Activation modules
# ======================================================================

class ReLU(Module):
    def forward(self, x): return F.relu(x)
    def __repr__(self): return "ReLU()"

class LeakyReLU(Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    def forward(self, x): return F.leaky_relu(x, self.negative_slope)
    def __repr__(self): return f"LeakyReLU({self.negative_slope})"

class ELU(Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x): return F.elu(x, self.alpha)

class Sigmoid(Module):
    def forward(self, x): return F.sigmoid(x)
    def __repr__(self): return "Sigmoid()"

class Tanh(Module):
    def forward(self, x): return F.tanh(x)
    def __repr__(self): return "Tanh()"

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    def forward(self, x): return F.softmax(x, self.dim)

class GELU(Module):
    def forward(self, x): return F.gelu(x)
    def __repr__(self): return "GELU()"

class SiLU(Module):
    def forward(self, x): return F.silu(x)
    def __repr__(self): return "SiLU()"

class Mish(Module):
    def forward(self, x): return F.mish(x)


# ======================================================================
#  Dropout
# ======================================================================

class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)

    def __repr__(self): return f"Dropout(p={self.p})"


class Dropout2d(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        N, C, H, W = x.data.shape
        mask = (np.random.rand(N, C, 1, 1) > self.p).astype(x.data.dtype)
        return Tensor(x.data * mask / (1 - self.p))


# ======================================================================
#  Recurrent layers
# ======================================================================

class RNNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, nonlinearity: str = "tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        k = np.sqrt(1 / hidden_size)
        self.weight_ih = Parameter(np.random.uniform(-k, k, (hidden_size, input_size)))
        self.weight_hh = Parameter(np.random.uniform(-k, k, (hidden_size, hidden_size)))
        self.bias_ih = Parameter(np.zeros(hidden_size))
        self.bias_hh = Parameter(np.zeros(hidden_size))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        if h is None:
            h = Tensor(np.zeros((x.data.shape[0], self.hidden_size)))
        pre = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(h, self.weight_hh, self.bias_hh)
        if self.nonlinearity == "tanh":
            return F.tanh(pre)
        return F.relu(pre)


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = np.sqrt(1 / hidden_size)
        self.weight_ih = Parameter(np.random.uniform(-k, k, (4 * hidden_size, input_size)))
        self.weight_hh = Parameter(np.random.uniform(-k, k, (4 * hidden_size, hidden_size)))
        self.bias = Parameter(np.zeros(4 * hidden_size))

    def forward(self, x: Tensor, state: Optional[Tuple] = None) -> Tuple[Tensor, Tensor]:
        h = self.hidden_size
        if state is None:
            hx = Tensor(np.zeros((x.data.shape[0], h)))
            cx = Tensor(np.zeros((x.data.shape[0], h)))
        else:
            hx, cx = state
        gates = F.linear(x, self.weight_ih) + F.linear(hx, self.weight_hh) + self.bias
        i_gate = F.sigmoid(Tensor(gates.data[:, :h]))
        f_gate = F.sigmoid(Tensor(gates.data[:, h:2*h]))
        g_gate = F.tanh(Tensor(gates.data[:, 2*h:3*h]))
        o_gate = F.sigmoid(Tensor(gates.data[:, 3*h:]))
        cy = Tensor(f_gate.data * cx.data + i_gate.data * g_gate.data)
        hy = Tensor(o_gate.data * np.tanh(cy.data))
        return hy, cy


class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        k = np.sqrt(1 / hidden_size)
        self.weight_ih = Parameter(np.random.uniform(-k, k, (3 * hidden_size, input_size)))
        self.weight_hh = Parameter(np.random.uniform(-k, k, (3 * hidden_size, hidden_size)))
        self.bias = Parameter(np.zeros(3 * hidden_size))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tensor:
        hs = self.hidden_size
        if h is None:
            h = Tensor(np.zeros((x.data.shape[0], hs)))
        gi = x.data @ self.weight_ih.data.T + self.bias.data
        gh = h.data @ self.weight_hh.data.T
        r = F.sigmoid(Tensor(gi[:, :hs] + gh[:, :hs]))
        z = F.sigmoid(Tensor(gi[:, hs:2*hs] + gh[:, hs:2*hs]))
        n = F.tanh(Tensor(gi[:, 2*hs:] + r.data * gh[:, 2*hs:]))
        hy = Tensor((1 - z.data) * n.data + z.data * h.data)
        return hy


# ======================================================================
#  Attention
# ======================================================================

class MultiheadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.scale = self.head_dim ** -0.5

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        B, Sq, E = query.data.shape
        _, Sk, _ = key.data.shape
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(query).data.reshape(B, Sq, H, D).transpose(0, 2, 1, 3)
        K = self.k_proj(key).data.reshape(B, Sk, H, D).transpose(0, 2, 1, 3)
        V = self.v_proj(value).data.reshape(B, Sk, H, D).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) * self.scale
        if mask is not None:
            scores = scores + mask.data
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        if self.training and self.dropout_p > 0:
            attn = attn * (np.random.rand(*attn.shape) > self.dropout_p) / (1 - self.dropout_p)

        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, Sq, E)
        out = self.out_proj(Tensor(out))
        return out, Tensor(attn.mean(axis=1))


# ======================================================================
#  Containers
# ======================================================================

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for i, m in enumerate(args):
            self._modules[str(i)] = m
            object.__setattr__(self, str(i), m)

    def forward(self, x: Tensor) -> Tensor:
        for m in self._modules.values():
            x = m(x)
        return x

    def __repr__(self):
        lines = ["Sequential("]
        for i, m in self._modules.items():
            lines.append(f"  ({i}): {repr(m)}")
        lines.append(")")
        return "\n".join(lines)

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]


class ModuleList(Module):
    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules:
            for i, m in enumerate(modules):
                self._modules[str(i)] = m

    def append(self, m: Module) -> "ModuleList":
        self._modules[str(len(self._modules))] = m
        return self

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())


class ModuleDict(Module):
    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        if modules:
            for k, m in modules.items():
                self._modules[k] = m

    def __getitem__(self, key):
        return self._modules[key]

    def __setitem__(self, key, m):
        self._modules[key] = m

    def keys(self): return self._modules.keys()
    def values(self): return self._modules.values()
    def items(self): return self._modules.items()


# ======================================================================
#  Identity / utility layers
# ======================================================================

class Identity(Module):
    def forward(self, x): return x
    def __repr__(self): return "Identity()"


class Upsample(Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        s = self.scale_factor
        return Tensor(np.repeat(np.repeat(x.data, s, axis=2), s, axis=3))


# Export loss functions as classes
class MSELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        return F.mse_loss(pred, target, self.reduction)

class CrossEntropyLoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, logits, targets):
        return F.cross_entropy_loss(logits, targets, self.reduction)

class BCELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        return F.binary_cross_entropy(pred, target, self.reduction)

class HuberLoss(Module):
    def __init__(self, delta=1.0, reduction="mean"):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    def forward(self, pred, target):
        return F.huber_loss(pred, target, self.delta, self.reduction)

class MAELoss(Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred, target):
        return F.mae_loss(pred, target, self.reduction)
