"""
TensorEra
=========
A modern, productive deep learning framework — inspired by PyTorch,
built on NumPy. Runs on any device without installation hell.

Quick start::

    import tensorera as te

    x = te.randn(3, 4)
    y = te.randn(4, 5)
    z = x @ y
    print(z)

Device support::

    device = te.Device.auto_select()  # Picks best available device
    x = te.randn(3, 4, device="cpu")
    x = x.cuda()    # Move to CUDA if available
    x = x.cpu()     # Back to CPU

Training loop::

    model = te.nn.Sequential(
        te.nn.Linear(784, 256),
        te.nn.ReLU(),
        te.nn.Linear(256, 10),
    )
    optimizer = te.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = te.nn.CrossEntropyLoss()

    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
"""

__version__ = "1.0.0"
__author__ = "TensorEra Contributors"

from .tensor import Tensor, Device
from .ops import (
    # Creation
    tensor, zeros, ones, zeros_like, ones_like, full, full_like,
    eye, arange, linspace, logspace,
    # Random
    rand, randn, randint, rand_like, randn_like, manual_seed,
    # Math
    matmul, dot, outer, cross,
    exp, log, log2, log10, sqrt, abs, sign,
    ceil, floor, round,
    sin, cos, tan, arcsin, arccos, arctan, arctan2,
    sigmoid, tanh,
    # Manipulation
    cat, stack, split, squeeze, unsqueeze, flatten,
    transpose, permute, reshape, broadcast_to, where,
    unique, sort, topk, argmax, argmin, einsum, norm,
    # Persistence
    save, load,
    # Context managers
    no_grad, enable_grad,
)
from . import functional as F
from . import nn
from . import optim
from . import data


# Convenience aliases
from .functional import (
    relu, leaky_relu, elu, gelu, silu, mish, hardswish,
    softmax, log_softmax,
    mse_loss, mae_loss, cross_entropy_loss, binary_cross_entropy,
    huber_loss, kl_div_loss,
    dropout, linear as functional_linear,
    concatenate, clip_grad_norm_,
    one_hot, batch_norm, layer_norm,
)

from .data import (
    Dataset, TensorDataset, Subset, ConcatDataset,
    DataLoader, Sampler, RandomSampler, SequentialSampler,
    WeightedRandomSampler, BatchSampler,
    Compose, Normalize, ToTensor,
    RandomHorizontalFlip, RandomVerticalFlip,
    RandomCrop, CenterCrop, GaussianNoise,
    random_split,
)


# ======================================================================
#  Convenience device helpers
# ======================================================================

def device(device_str: str) -> Device:
    return Device(device_str)


def is_available(device_str: str = "cuda") -> bool:
    return Device(device_str).is_available()


def cuda_is_available() -> bool:
    return is_available("cuda")


def get_device() -> Device:
    """Return best available device."""
    return Device.auto_select()


# ======================================================================
#  Version info
# ======================================================================

def version_info() -> str:
    import numpy as np
    import sys
    cuda_available = cuda_is_available()
    lines = [
        "=" * 45,
        f"  TensorEra v{__version__}",
        "=" * 45,
        f"  Python   : {sys.version.split()[0]}",
        f"  NumPy    : {np.__version__}",
        f"  CUDA     : {'Available ✓' if cuda_available else 'Not found (CPU mode)'}",
        "=" * 45,
    ]
    try:
        import cupy
        lines.insert(-1, f"  CuPy     : {cupy.__version__} ✓")
    except ImportError:
        pass
    return "\n".join(lines)


print(version_info())
