"""
TensorEra - Optimizers
PyTorch-compatible optimizer implementations.
"""

import numpy as np
from typing import List, Optional, Iterable, Dict
from .tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Iterable, defaults: Dict):
        self.param_groups = [{"params": list(params), **defaults}]
        self.defaults = defaults
        self._step_count = 0

    def zero_grad(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = None

    def step(self):
        raise NotImplementedError

    def add_param_group(self, param_group: Dict):
        self.param_groups.append({**self.defaults, **param_group})

    @property
    def state_dict(self):
        return {
            "step": self._step_count,
            "param_groups": self.param_groups,
        }


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    Supports momentum, weight decay, and Nesterov.
    """

    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0,
                 weight_decay: float = 0.0, nesterov: bool = False, dampening: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum,
                                  "weight_decay": weight_decay, "nesterov": nesterov,
                                  "dampening": dampening})
        self._velocity: Dict[int, np.ndarray] = {}

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            dampening = group["dampening"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data.copy()
                if wd != 0:
                    g += wd * p.data
                if mu != 0:
                    pid = id(p)
                    if pid not in self._velocity:
                        self._velocity[pid] = g.copy()
                    else:
                        self._velocity[pid] = mu * self._velocity[pid] + (1 - dampening) * g
                    if nesterov:
                        g = g + mu * self._velocity[pid]
                    else:
                        g = self._velocity[pid]
                p.data -= lr * g
        self._step_count += 1


class Adam(Optimizer):
    """
    Adam optimizer: Adaptive Moment Estimation.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0, amsgrad: bool = False):
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps,
                                  "weight_decay": weight_decay, "amsgrad": amsgrad})
        self._m: Dict[int, np.ndarray] = {}
        self._v: Dict[int, np.ndarray] = {}
        self._v_max: Dict[int, np.ndarray] = {}
        self._t: Dict[int, int] = {}

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad.data.copy()
                if wd != 0:
                    g += wd * p.data

                self._t[pid] = self._t.get(pid, 0) + 1
                t = self._t[pid]

                if pid not in self._m:
                    self._m[pid] = np.zeros_like(g)
                    self._v[pid] = np.zeros_like(g)
                    if amsgrad:
                        self._v_max[pid] = np.zeros_like(g)

                self._m[pid] = b1 * self._m[pid] + (1 - b1) * g
                self._v[pid] = b2 * self._v[pid] + (1 - b2) * g ** 2

                m_hat = self._m[pid] / (1 - b1 ** t)
                v_hat = self._v[pid] / (1 - b2 ** t)

                if amsgrad:
                    self._v_max[pid] = np.maximum(self._v_max[pid], v_hat)
                    denom = np.sqrt(self._v_max[pid]) + eps
                else:
                    denom = np.sqrt(v_hat) + eps

                p.data -= lr * m_hat / denom
        self._step_count += 1


class AdamW(Adam):
    """
    AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
    """

    def step(self):
        for group in self.param_groups:
            wd = group["weight_decay"]
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Decoupled weight decay
                if wd != 0:
                    p.data *= (1 - lr * wd)
                # Remove wd from gradient computation
                orig_wd = group["weight_decay"]
                group["weight_decay"] = 0
                super_step = True
        # Call Adam step without weight decay
        for group in self.param_groups:
            group["weight_decay"] = 0
        super().step()
        # Restore
        for group in self.param_groups:
            group["weight_decay"] = orig_wd if "orig_wd" in dir() else group.get("weight_decay", 0)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, params, lr: float = 1e-2, alpha: float = 0.99,
                 eps: float = 1e-8, weight_decay: float = 0.0, momentum: float = 0.0):
        super().__init__(params, {"lr": lr, "alpha": alpha, "eps": eps,
                                  "weight_decay": weight_decay, "momentum": momentum})
        self._sq: Dict[int, np.ndarray] = {}
        self._buf: Dict[int, np.ndarray] = {}

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            wd = group["weight_decay"]
            mu = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad.data.copy()
                if wd != 0:
                    g += wd * p.data

                if pid not in self._sq:
                    self._sq[pid] = np.zeros_like(g)

                self._sq[pid] = alpha * self._sq[pid] + (1 - alpha) * g ** 2
                avg = np.sqrt(self._sq[pid]) + eps

                if mu != 0:
                    if pid not in self._buf:
                        self._buf[pid] = np.zeros_like(g)
                    self._buf[pid] = mu * self._buf[pid] + g / avg
                    p.data -= lr * self._buf[pid]
                else:
                    p.data -= lr * g / avg
        self._step_count += 1


class Adagrad(Optimizer):
    """Adagrad optimizer."""

    def __init__(self, params, lr: float = 1e-2, eps: float = 1e-10, weight_decay: float = 0.0):
        super().__init__(params, {"lr": lr, "eps": eps, "weight_decay": weight_decay})
        self._sum: Dict[int, np.ndarray] = {}

    def step(self):
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad.data.copy()
                if wd != 0:
                    g += wd * p.data
                if pid not in self._sum:
                    self._sum[pid] = np.zeros_like(g)
                self._sum[pid] += g ** 2
                p.data -= lr * g / (np.sqrt(self._sum[pid]) + eps)
        self._step_count += 1


class Adadelta(Optimizer):
    """Adadelta optimizer."""

    def __init__(self, params, lr: float = 1.0, rho: float = 0.9, eps: float = 1e-6):
        super().__init__(params, {"lr": lr, "rho": rho, "eps": eps})
        self._eg2: Dict[int, np.ndarray] = {}
        self._ex2: Dict[int, np.ndarray] = {}

    def step(self):
        for group in self.param_groups:
            lr, rho, eps = group["lr"], group["rho"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                g = p.grad.data.copy()
                if pid not in self._eg2:
                    self._eg2[pid] = np.zeros_like(g)
                    self._ex2[pid] = np.zeros_like(g)
                self._eg2[pid] = rho * self._eg2[pid] + (1 - rho) * g ** 2
                dx = np.sqrt(self._ex2[pid] + eps) / np.sqrt(self._eg2[pid] + eps) * g
                self._ex2[pid] = rho * self._ex2[pid] + (1 - rho) * dx ** 2
                p.data -= lr * dx
        self._step_count += 1


class LBFGS(Optimizer):
    """Limited-memory BFGS (simplified version)."""

    def __init__(self, params, lr: float = 1.0, max_iter: int = 20, history_size: int = 100):
        super().__init__(params, {"lr": lr, "max_iter": max_iter})

    def step(self, closure=None):
        # Fallback: gradient descent step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= group["lr"] * p.grad.data
        self._step_count += 1


# ======================================================================
#  Learning rate schedulers
# ======================================================================

class LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step()

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def step(self):
        self.last_epoch += 1
        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * (self.gamma ** (self.last_epoch // self.step_size))
                for base in self.base_lrs]


class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = self.gamma ** sum(1 for m in self.milestones if self.last_epoch >= m)
        return [base * factor for base in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base * (self.gamma ** self.last_epoch) for base in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.eta_min + (base - self.eta_min) *
                (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
                for base in self.base_lrs]


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""

    def __init__(self, optimizer: Optimizer, mode: str = "min", factor: float = 0.1,
                 patience: int = 10, min_lr: float = 0.0, verbose: bool = False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0

    def step(self, metric: float):
        improved = metric < self.best if self.mode == "min" else metric > self.best
        if improved:
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                for group in self.optimizer.param_groups:
                    new_lr = max(group["lr"] * self.factor, self.min_lr)
                    if self.verbose and new_lr != group["lr"]:
                        print(f"Reducing LR to {new_lr:.6f}")
                    group["lr"] = new_lr
                self.num_bad_epochs = 0


class OneCycleLR(LRScheduler):
    """1-cycle learning rate policy."""

    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        phase1_end = int(self.total_steps * self.pct_start)
        t = self.last_epoch
        if t <= phase1_end:
            progress = t / phase1_end
            lr = self.max_lr / self.div_factor + progress * (self.max_lr - self.max_lr / self.div_factor)
        else:
            progress = (t - phase1_end) / (self.total_steps - phase1_end)
            min_lr = self.max_lr / (self.div_factor * self.final_div_factor)
            lr = self.max_lr + progress * (min_lr - self.max_lr)
        return [lr for _ in self.base_lrs]


class WarmupScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            factor = self._step / self.warmup_steps
        else:
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            factor = max(0.0, 0.5 * (1 + np.cos(np.pi * progress)))
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = self.min_lr + factor * (base - self.min_lr)
