"""
TensorEra - Functional API
Collection of stateless functions for tensor operations.
"""

import numpy as np
from .tensor import Tensor, _accum_grad
from typing import Optional, List


# ======================================================================
#  Activation functions
# ======================================================================

def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        def _backward(grad):
            _accum_grad(x, Tensor(grad.data * (x.data > 0).astype(grad.data.dtype)))
        out._grad_fn = _backward
        out._children = [x]
    return out


def leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    mask = x.data > 0
    out_data = np.where(mask, x.data, negative_slope * x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        def _backward(grad):
            slope = np.where(mask, 1.0, negative_slope)
            _accum_grad(x, Tensor(grad.data * slope))
        out._grad_fn = _backward
        out._children = [x]
    return out


def elu(x: Tensor, alpha: float = 1.0) -> Tensor:
    out_data = np.where(x.data > 0, x.data, alpha * (np.exp(x.data) - 1))
    return Tensor(out_data)


def sigmoid(x: Tensor) -> Tensor:
    sig = 1 / (1 + np.exp(-x.data))
    out = Tensor(sig, requires_grad=x.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        def _backward(grad):
            _accum_grad(x, Tensor(grad.data * sig * (1 - sig)))
        out._grad_fn = _backward
        out._children = [x]
    return out


def tanh(x: Tensor) -> Tensor:
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        def _backward(grad):
            _accum_grad(x, Tensor(grad.data * (1 - t ** 2)))
        out._grad_fn = _backward
        out._children = [x]
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    x_max = x.data.max(axis=axis, keepdims=True)
    e = np.exp(x.data - x_max)
    s = e / e.sum(axis=axis, keepdims=True)
    return Tensor(s)


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    x_max = x.data.max(axis=axis, keepdims=True)
    log_sum = np.log(np.exp(x.data - x_max).sum(axis=axis, keepdims=True))
    return Tensor(x.data - x_max - log_sum)


def gelu(x: Tensor) -> Tensor:
    """Gaussian Error Linear Unit."""
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x.data + 0.044715 * x.data ** 3)))
    return Tensor(x.data * cdf)


def silu(x: Tensor) -> Tensor:
    """Sigmoid Linear Unit (Swish)."""
    sig = 1 / (1 + np.exp(-x.data))
    return Tensor(x.data * sig)


def mish(x: Tensor) -> Tensor:
    return Tensor(x.data * np.tanh(np.log1p(np.exp(x.data))))


def hardswish(x: Tensor) -> Tensor:
    return Tensor(x.data * np.clip(x.data + 3, 0, 6) / 6)


# ======================================================================
#  Loss functions
# ======================================================================

def mse_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = pred.data - target.data
    loss_data = diff ** 2
    if reduction == "mean":
        loss_data = loss_data.mean()
    elif reduction == "sum":
        loss_data = loss_data.sum()

    out = Tensor(loss_data, requires_grad=pred.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        n = pred.data.size if reduction == "mean" else 1
        def _backward(grad):
            _accum_grad(pred, Tensor(2 * diff / n * grad.data))
        out._grad_fn = _backward
        out._children = [pred]
    return out


def mae_loss(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    diff = np.abs(pred.data - target.data)
    if reduction == "mean":
        return Tensor(diff.mean())
    return Tensor(diff.sum())


def cross_entropy_loss(logits: Tensor, targets: Tensor, reduction: str = "mean") -> Tensor:
    """Cross-entropy loss from raw logits (no softmax needed)."""
    # Numerical stable
    x = logits.data
    t = targets.data.astype(int)
    x_max = x.max(axis=-1, keepdims=True)
    log_sum_exp = np.log(np.exp(x - x_max).sum(axis=-1)) + x_max.squeeze(-1)
    log_probs = x[np.arange(len(t)), t] - log_sum_exp
    loss = -log_probs

    if reduction == "mean":
        loss_val = loss.mean()
    elif reduction == "sum":
        loss_val = loss.sum()
    else:
        loss_val = loss

    out = Tensor(loss_val, requires_grad=logits.requires_grad)
    out._is_leaf = False
    if out.requires_grad:
        def _backward(grad):
            sm = np.exp(x - x.max(axis=-1, keepdims=True))
            sm /= sm.sum(axis=-1, keepdims=True)
            d = sm.copy()
            d[np.arange(len(t)), t] -= 1
            n = len(t) if reduction == "mean" else 1
            _accum_grad(logits, Tensor(d / n * grad.data))
        out._grad_fn = _backward
        out._children = [logits]
    return out


def binary_cross_entropy(pred: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    eps = 1e-12
    p = np.clip(pred.data, eps, 1 - eps)
    t = target.data
    loss = -(t * np.log(p) + (1 - t) * np.log(1 - p))
    if reduction == "mean":
        return Tensor(loss.mean())
    return Tensor(loss.sum())


def huber_loss(pred: Tensor, target: Tensor, delta: float = 1.0, reduction: str = "mean") -> Tensor:
    diff = np.abs(pred.data - target.data)
    loss = np.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))
    if reduction == "mean":
        return Tensor(loss.mean())
    return Tensor(loss.sum())


def kl_div_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    eps = 1e-12
    t = np.clip(target.data, eps, 1)
    i = np.clip(input.data, eps, 1)
    loss = t * (np.log(t) - np.log(i))
    if reduction == "mean":
        return Tensor(loss.mean())
    return Tensor(loss.sum())


# ======================================================================
#  Normalization
# ======================================================================

def batch_norm(x: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None,
               eps: float = 1e-5, training: bool = True) -> Tensor:
    if training:
        mean = x.data.mean(axis=(0, 2, 3) if x.ndim == 4 else 0, keepdims=True)
        var = x.data.var(axis=(0, 2, 3) if x.ndim == 4 else 0, keepdims=True)
    out = (x.data - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma.data
    if beta is not None:
        out = out + beta.data
    return Tensor(out)


def layer_norm(x: Tensor, normalized_shape, gamma: Optional[Tensor] = None,
               beta: Optional[Tensor] = None, eps: float = 1e-5) -> Tensor:
    axes = tuple(range(-len(normalized_shape), 0))
    mean = x.data.mean(axis=axes, keepdims=True)
    var = x.data.var(axis=axes, keepdims=True)
    out = (x.data - mean) / np.sqrt(var + eps)
    if gamma is not None:
        out = out * gamma.data
    if beta is not None:
        out = out + beta.data
    return Tensor(out)


# ======================================================================
#  Tensor utilities
# ======================================================================

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not training or p == 0:
        return x
    mask = (np.random.rand(*x.data.shape) > p).astype(x.data.dtype)
    return Tensor(x.data * mask / (1 - p))


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    out = x @ weight.T
    if bias is not None:
        out = out + bias
    return out


def pad(x: Tensor, pad_width, mode: str = "constant", value: float = 0) -> Tensor:
    if mode == "constant":
        return Tensor(np.pad(x.data, pad_width, mode=mode, constant_values=value))
    return Tensor(np.pad(x.data, pad_width, mode=mode))


def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    return Tensor(np.concatenate([t.data for t in tensors], axis=axis))


def stack(tensors: List[Tensor], axis: int = 0) -> Tensor:
    return Tensor(np.stack([t.data for t in tensors], axis=axis))


def split(x: Tensor, indices_or_sections, axis: int = 0) -> List[Tensor]:
    return [Tensor(a) for a in np.split(x.data, indices_or_sections, axis=axis)]


def one_hot(indices: Tensor, num_classes: int) -> Tensor:
    idx = indices.data.astype(int).flatten()
    out = np.zeros((len(idx), num_classes))
    out[np.arange(len(idx)), idx] = 1
    return Tensor(out)


def argmax(x: Tensor, axis: Optional[int] = None) -> Tensor:
    return Tensor(np.argmax(x.data, axis=axis))


def argmin(x: Tensor, axis: Optional[int] = None) -> Tensor:
    return Tensor(np.argmin(x.data, axis=axis))


def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
    return Tensor(np.where(condition.data, x.data, y.data))


def einsum(equation: str, *operands) -> Tensor:
    return Tensor(np.einsum(equation, *[op.data for op in operands]))


def norm(x: Tensor, ord=None, axis=None) -> Tensor:
    return Tensor(np.linalg.norm(x.data, ord=ord, axis=axis))


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradients of an iterable of parameters by norm."""
    grads = [p.grad.data for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data *= clip_coef
    return float(total_norm)
