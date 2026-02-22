"""
TensorEra - Core Tensor Module
Provides multi-dimensional array operations with autograd support.
"""

import numpy as np
import math
from typing import Optional, Union, List, Tuple, Callable


class Device:
    """Represents a compute device."""
    
    BACKENDS = {}
    
    def __init__(self, device_str: str = "cpu"):
        device_str = device_str.lower().strip()
        if ":" in device_str:
            self.type, idx = device_str.split(":")
            self.index = int(idx)
        else:
            self.type = device_str
            self.index = 0
        self._validate()

    def _validate(self):
        valid = ["cpu", "cuda", "mps", "tpu", "rocm", "auto"]
        if self.type not in valid:
            raise ValueError(f"Unknown device '{self.type}'. Available: {valid}")

    def __repr__(self):
        if self.index == 0:
            return f"device('{self.type}')"
        return f"device('{self.type}:{self.index}')"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.type == other
        return self.type == other.type and self.index == other.index

    @staticmethod
    def auto_select():
        """Automatically selects the best available device."""
        try:
            import cupy
            return Device("cuda")
        except ImportError:
            pass
        try:
            import torch
            if torch.backends.mps.is_available():
                return Device("mps")
        except ImportError:
            pass
        return Device("cpu")

    def is_available(self) -> bool:
        if self.type == "cpu":
            return True
        if self.type == "cuda":
            try:
                import cupy
                return cupy.cuda.is_available()
            except ImportError:
                return False
        if self.type == "mps":
            try:
                import torch
                return torch.backends.mps.is_available()
            except ImportError:
                return False
        return False


def _to_numpy(data):
    if isinstance(data, Tensor):
        return data.data
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


class Tensor:
    """
    Multi-dimensional array with automatic differentiation support.
    Core data structure of TensorEra.
    """

    def __init__(
        self,
        data,
        dtype=None,
        device: Union[str, Device] = "cpu",
        requires_grad: bool = False,
        name: Optional[str] = None,
    ):
        if isinstance(data, Tensor):
            data = data.data
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype)
        elif dtype is not None:
            data = data.astype(dtype)

        self.data = data
        self.dtype = self.data.dtype
        self.device = Device(device) if isinstance(device, str) else device
        self.requires_grad = requires_grad
        self.name = name

        # Autograd internals
        self.grad: Optional["Tensor"] = None
        self._grad_fn: Optional[Callable] = None
        self._children: List["Tensor"] = []
        self._is_leaf = True

    # ------------------------------------------------------------------ #
    #  Shape / dtype properties
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    # ------------------------------------------------------------------ #
    #  Device management
    # ------------------------------------------------------------------ #
    def to(self, device: Union[str, Device]) -> "Tensor":
        """Move tensor to device (CPU only in pure NumPy backend)."""
        new_device = Device(device) if isinstance(device, str) else device
        t = Tensor(self.data.copy(), device=new_device, requires_grad=self.requires_grad)
        return t

    def cuda(self, index: int = 0) -> "Tensor":
        return self.to(f"cuda:{index}")

    def cpu(self) -> "Tensor":
        return self.to("cpu")

    # ------------------------------------------------------------------ #
    #  Numpy interop
    # ------------------------------------------------------------------ #
    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError("Call .detach().numpy() on a tensor that requires grad")
        return self.data

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), dtype=self.dtype, device=self.device)

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    # ------------------------------------------------------------------ #
    #  Arithmetic operations
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    g = grad.data
                    # Handle broadcasting
                    while g.ndim > self.data.ndim:
                        g = g.sum(axis=0)
                    for i, (ds, do) in enumerate(zip(self.data.shape, g.shape)):
                        if ds == 1 and do > 1:
                            g = g.sum(axis=i, keepdims=True)
                    _accum_grad(self, Tensor(g))
                if other.requires_grad:
                    g = grad.data
                    while g.ndim > other.data.ndim:
                        g = g.sum(axis=0)
                    for i, (ds, do) in enumerate(zip(other.data.shape, g.shape)):
                        if ds == 1 and do > 1:
                            g = g.sum(axis=i, keepdims=True)
                    _accum_grad(other, Tensor(g))
            out._grad_fn = _backward
            out._children = [self, other]
        return out

    def __radd__(self, other): return self.__add__(other)

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    _accum_grad(self, Tensor(grad.data * other.data))
                if other.requires_grad:
                    _accum_grad(other, Tensor(grad.data * self.data))
            out._grad_fn = _backward
            out._children = [self, other]
        return out

    def __rmul__(self, other): return self.__mul__(other)

    def __sub__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.__add__(other.__neg__())

    def __rsub__(self, other):
        return Tensor(other).__sub__(self)

    def __neg__(self) -> "Tensor":
        return self.__mul__(Tensor(-1.0))

    def __truediv__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.__mul__(other.pow(-1))

    def __rtruediv__(self, other):
        return Tensor(other).__truediv__(self)

    def __matmul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    _accum_grad(self, Tensor(grad.data @ other.data.T))
                if other.requires_grad:
                    _accum_grad(other, Tensor(self.data.T @ grad.data))
            out._grad_fn = _backward
            out._children = [self, other]
        return out

    def __pow__(self, exp) -> "Tensor":
        return self.pow(exp)

    def pow(self, exp) -> "Tensor":
        out = Tensor(self.data ** exp, requires_grad=self.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                _accum_grad(self, Tensor(exp * (self.data ** (exp - 1)) * grad.data))
            out._grad_fn = _backward
            out._children = [self]
        return out

    # ------------------------------------------------------------------ #
    #  Reduction operations
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                g = grad.data
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                _accum_grad(self, Tensor(np.broadcast_to(g, self.data.shape).copy()))
            out._grad_fn = _backward
            out._children = [self]
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        s = self.sum(axis=axis, keepdims=keepdims)
        return Tensor(s.data / n, requires_grad=s.requires_grad)

    def max(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        return out

    def min(self, axis=None, keepdims=False) -> "Tensor":
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)

    def var(self, axis=None, keepdims=False) -> "Tensor":
        return Tensor(self.data.var(axis=axis, keepdims=keepdims))

    def std(self, axis=None, keepdims=False) -> "Tensor":
        return Tensor(self.data.std(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------------ #
    #  Shape operations
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                _accum_grad(self, Tensor(grad.data.reshape(self.data.shape)))
            out._grad_fn = _backward
            out._children = [self]
        return out

    def view(self, *shape) -> "Tensor":
        return self.reshape(*shape)

    def squeeze(self, axis=None) -> "Tensor":
        return Tensor(self.data.squeeze(axis=axis), requires_grad=self.requires_grad)

    def unsqueeze(self, axis: int) -> "Tensor":
        return Tensor(np.expand_dims(self.data, axis=axis), requires_grad=self.requires_grad)

    def flatten(self) -> "Tensor":
        return self.reshape(-1)

    def transpose(self, axes=None) -> "Tensor":
        return Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)

    def permute(self, *dims) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = dims[0]
        return Tensor(self.data.transpose(dims), requires_grad=self.requires_grad)

    def contiguous(self) -> "Tensor":
        return Tensor(np.ascontiguousarray(self.data), requires_grad=self.requires_grad)

    # ------------------------------------------------------------------ #
    #  Math operations
    # ------------------------------------------------------------------ #
    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                _accum_grad(self, Tensor(grad.data * out_data))
            out._grad_fn = _backward
            out._children = [self]
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data + 1e-12), requires_grad=self.requires_grad)
        out._is_leaf = False
        if out.requires_grad:
            def _backward(grad):
                _accum_grad(self, Tensor(grad.data / (self.data + 1e-12)))
            out._grad_fn = _backward
            out._children = [self]
        return out

    def sqrt(self) -> "Tensor":
        return self.pow(0.5)

    def abs(self) -> "Tensor":
        return Tensor(np.abs(self.data))

    def clip(self, min_val=None, max_val=None) -> "Tensor":
        return Tensor(np.clip(self.data, min_val, max_val))

    def clamp(self, min_val=None, max_val=None) -> "Tensor":
        return self.clip(min_val, max_val)

    # ------------------------------------------------------------------ #
    #  Autograd
    # ------------------------------------------------------------------ #
    def backward(self, gradient: Optional["Tensor"] = None):
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")
        if gradient is None:
            if self.data.size != 1:
                raise RuntimeError("backward() on non-scalar requires gradient argument")
            gradient = Tensor(np.ones_like(self.data))

        # Topological sort
        topo = []
        visited = set()

        def build(t):
            if id(t) not in visited:
                visited.add(id(t))
                for child in t._children:
                    build(child)
                topo.append(t)

        build(self)
        self.grad = gradient

        for t in reversed(topo):
            if t._grad_fn and t.grad is not None:
                t._grad_fn(t.grad)

    def zero_grad(self):
        self.grad = None

    # ------------------------------------------------------------------ #
    #  Comparison
    # ------------------------------------------------------------------ #
    def __eq__(self, other):
        other_data = _to_numpy(other)
        return Tensor(self.data == other_data)

    def __lt__(self, other):
        return Tensor(self.data < _to_numpy(other))

    def __le__(self, other):
        return Tensor(self.data <= _to_numpy(other))

    def __gt__(self, other):
        return Tensor(self.data > _to_numpy(other))

    def __ge__(self, other):
        return Tensor(self.data >= _to_numpy(other))

    # ------------------------------------------------------------------ #
    #  Indexing
    # ------------------------------------------------------------------ #
    def __getitem__(self, idx):
        return Tensor(self.data[idx], requires_grad=self.requires_grad)

    def __setitem__(self, idx, val):
        self.data[idx] = _to_numpy(val)

    # ------------------------------------------------------------------ #
    #  Display
    # ------------------------------------------------------------------ #
    def __repr__(self):
        grad_str = ", requires_grad=True" if self.requires_grad else ""
        dev_str = f", device='{self.device.type}'" if self.device.type != "cpu" else ""
        return f"tensor({self.data}{grad_str}{dev_str})"

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __float__(self):
        return float(self.data)

    def __int__(self):
        return int(self.data)


def _accum_grad(tensor: Tensor, grad: Tensor):
    """Accumulate gradient into tensor."""
    if tensor.grad is None:
        tensor.grad = Tensor(grad.data.copy())
    else:
        tensor.grad.data += grad.data
