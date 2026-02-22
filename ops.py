"""
TensorEra - Tensor creation and utility operations.
"""

import numpy as np
from typing import Union, Tuple, Optional, List
from .tensor import Tensor, Device


# ======================================================================
#  Tensor creation
# ======================================================================

def tensor(data, dtype=None, device="cpu", requires_grad=False) -> Tensor:
    return Tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(*shape, dtype=np.float32, device="cpu", requires_grad=False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.zeros(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def ones(*shape, dtype=np.float32, device="cpu", requires_grad=False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.ones(shape, dtype=dtype), device=device, requires_grad=requires_grad)


def zeros_like(t: Tensor, dtype=None, device=None) -> Tensor:
    d = t.device if device is None else Device(device)
    dt = t.dtype if dtype is None else dtype
    return Tensor(np.zeros_like(t.data, dtype=dt), device=d)


def ones_like(t: Tensor, dtype=None, device=None) -> Tensor:
    d = t.device if device is None else Device(device)
    dt = t.dtype if dtype is None else dtype
    return Tensor(np.ones_like(t.data, dtype=dt), device=d)


def full(shape, fill_value, dtype=np.float32, device="cpu", requires_grad=False) -> Tensor:
    return Tensor(np.full(shape, fill_value, dtype=dtype), device=device, requires_grad=requires_grad)


def full_like(t: Tensor, fill_value) -> Tensor:
    return Tensor(np.full_like(t.data, fill_value))


def eye(n: int, m: Optional[int] = None, dtype=np.float32, device="cpu") -> Tensor:
    return Tensor(np.eye(n, m, dtype=dtype), device=device)


def arange(start, stop=None, step=1, dtype=None, device="cpu") -> Tensor:
    if stop is None:
        start, stop = 0, start
    return Tensor(np.arange(start, stop, step, dtype=dtype), device=device)


def linspace(start, end, steps: int, device="cpu") -> Tensor:
    return Tensor(np.linspace(start, end, steps), device=device)


def logspace(start, end, steps: int, base: float = 10.0, device="cpu") -> Tensor:
    return Tensor(np.logspace(start, end, steps, base=base), device=device)


# ======================================================================
#  Random tensors
# ======================================================================

def rand(*shape, dtype=np.float32, device="cpu", requires_grad=False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.rand(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def randn(*shape, dtype=np.float32, device="cpu", requires_grad=False) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return Tensor(np.random.randn(*shape).astype(dtype), device=device, requires_grad=requires_grad)


def randint(low, high=None, size=(1,), dtype=np.int64, device="cpu") -> Tensor:
    if high is None:
        low, high = 0, low
    return Tensor(np.random.randint(low, high, size=size, dtype=dtype), device=device)


def rand_like(t: Tensor) -> Tensor:
    return Tensor(np.random.rand(*t.shape).astype(t.dtype), device=t.device)


def randn_like(t: Tensor) -> Tensor:
    return Tensor(np.random.randn(*t.shape).astype(t.dtype), device=t.device)


def manual_seed(seed: int):
    np.random.seed(seed)


# ======================================================================
#  Math functions
# ======================================================================

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def dot(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.dot(a.data, b.data))


def outer(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(np.outer(a.data, b.data))


def cross(a: Tensor, b: Tensor, axis: int = -1) -> Tensor:
    return Tensor(np.cross(a.data, b.data, axis=axis))


def exp(x: Tensor) -> Tensor:
    return x.exp()


def log(x: Tensor) -> Tensor:
    return x.log()


def log2(x: Tensor) -> Tensor:
    return Tensor(np.log2(x.data + 1e-12))


def log10(x: Tensor) -> Tensor:
    return Tensor(np.log10(x.data + 1e-12))


def sqrt(x: Tensor) -> Tensor:
    return x.sqrt()


def abs(x: Tensor) -> Tensor:
    return x.abs()


def sign(x: Tensor) -> Tensor:
    return Tensor(np.sign(x.data))


def ceil(x: Tensor) -> Tensor:
    return Tensor(np.ceil(x.data))


def floor(x: Tensor) -> Tensor:
    return Tensor(np.floor(x.data))


def round(x: Tensor, decimals: int = 0) -> Tensor:
    return Tensor(np.round(x.data, decimals))


def sin(x: Tensor) -> Tensor:
    return Tensor(np.sin(x.data))


def cos(x: Tensor) -> Tensor:
    return Tensor(np.cos(x.data))


def tan(x: Tensor) -> Tensor:
    return Tensor(np.tan(x.data))


def arcsin(x: Tensor) -> Tensor:
    return Tensor(np.arcsin(x.data))


def arccos(x: Tensor) -> Tensor:
    return Tensor(np.arccos(x.data))


def arctan(x: Tensor) -> Tensor:
    return Tensor(np.arctan(x.data))


def arctan2(y: Tensor, x: Tensor) -> Tensor:
    return Tensor(np.arctan2(y.data, x.data))


def sigmoid(x: Tensor) -> Tensor:
    from . import functional as F
    return F.sigmoid(x)


def tanh(x: Tensor) -> Tensor:
    from . import functional as F
    return F.tanh(x)


# ======================================================================
#  Tensor manipulation
# ======================================================================

def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    from . import functional as F
    return F.concatenate(tensors, axis=dim)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    from . import functional as F
    return F.stack(tensors, axis=dim)


def split(x: Tensor, split_size_or_sections, dim: int = 0) -> List[Tensor]:
    from . import functional as F
    return F.split(x, split_size_or_sections, axis=dim)


def squeeze(x: Tensor, dim: Optional[int] = None) -> Tensor:
    return x.squeeze(dim)


def unsqueeze(x: Tensor, dim: int) -> Tensor:
    return x.unsqueeze(dim)


def flatten(x: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    shape = x.data.shape
    end = len(shape) if end_dim == -1 else end_dim + 1
    new_shape = shape[:start_dim] + (-1,) + shape[end:]
    return Tensor(x.data.reshape(new_shape))


def transpose(x: Tensor, dim0: int, dim1: int) -> Tensor:
    axes = list(range(x.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(x.data.transpose(axes))


def permute(x: Tensor, dims) -> Tensor:
    return Tensor(x.data.transpose(dims))


def reshape(x: Tensor, shape) -> Tensor:
    return x.reshape(shape)


def broadcast_to(x: Tensor, shape) -> Tensor:
    return Tensor(np.broadcast_to(x.data, shape))


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    from . import functional as F
    return F.where(condition, x, y)


def unique(x: Tensor, sorted: bool = True) -> Tensor:
    return Tensor(np.unique(x.data))


def sort(x: Tensor, dim: int = -1, descending: bool = False):
    idx = np.argsort(x.data, axis=dim)
    if descending:
        idx = idx[..., ::-1]
    return Tensor(np.take_along_axis(x.data, idx, axis=dim)), Tensor(idx)


def topk(x: Tensor, k: int, dim: int = -1, largest: bool = True):
    idx = np.argsort(x.data, axis=dim)
    if largest:
        idx = idx[..., -k:][:, ::-1]
    else:
        idx = idx[..., :k]
    values = np.take_along_axis(x.data, idx, axis=dim)
    return Tensor(values), Tensor(idx)


def argmax(x: Tensor, dim: Optional[int] = None) -> Tensor:
    return Tensor(np.argmax(x.data, axis=dim))


def argmin(x: Tensor, dim: Optional[int] = None) -> Tensor:
    return Tensor(np.argmin(x.data, axis=dim))


def einsum(equation: str, *operands: Tensor) -> Tensor:
    from . import functional as F
    return F.einsum(equation, *operands)


def norm(x: Tensor, p=2, dim: Optional[int] = None, keepdim: bool = False) -> Tensor:
    if p == 2:
        return Tensor(np.linalg.norm(x.data, ord=None, axis=dim, keepdims=keepdim))
    return Tensor(np.linalg.norm(x.data, ord=p, axis=dim, keepdims=keepdim))


# ======================================================================
#  Saving and loading
# ======================================================================

def save(obj, path: str):
    """Save a tensor, dict of tensors, or model state dict."""
    import os
    import pickle
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if isinstance(obj, Tensor):
        np.save(path, obj.data)
    elif isinstance(obj, dict):
        np.savez(path, **{k: v if isinstance(v, np.ndarray) else np.array(v)
                         for k, v in obj.items()})
    else:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def load(path: str):
    """Load saved tensor or dict."""
    import pickle
    if path.endswith(".npy"):
        return Tensor(np.load(path))
    if path.endswith(".npz"):
        data = np.load(path)
        return {k: Tensor(v) for k, v in data.items()}
    with open(path, "rb") as f:
        return pickle.load(f)


# ======================================================================
#  No-grad context manager
# ======================================================================

class no_grad:
    """Context manager to disable gradient computation."""
    _enabled = True

    def __enter__(self):
        self._prev = no_grad._enabled
        no_grad._enabled = False

    def __exit__(self, *args):
        no_grad._enabled = self._prev


class enable_grad:
    def __enter__(self):
        self._prev = no_grad._enabled
        no_grad._enabled = True

    def __exit__(self, *args):
        no_grad._enabled = self._prev
