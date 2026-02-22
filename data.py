"""
TensorEra - Data utilities
Dataset, DataLoader, transforms, and samplers.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Iterator, Union
from .tensor import Tensor


# ======================================================================
#  Datasets
# ======================================================================

class Dataset:
    """Abstract base class for datasets."""

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors. Each sample is retrieved by indexing tensors along the first dimension."""

    def __init__(self, *tensors: Tensor):
        assert all(t.data.shape[0] == tensors[0].data.shape[0] for t in tensors), \
            "All tensors must have the same first dimension size"
        self.tensors = tensors

    def __getitem__(self, idx):
        return tuple(Tensor(t.data[idx]) for t in self.tensors)

    def __len__(self):
        return self.tensors[0].data.shape[0]


class Subset(Dataset):
    """Subset of a dataset at specified indices."""

    def __init__(self, dataset: Dataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    """Dataset that is a concatenation of multiple datasets."""

    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self._lengths = [len(d) for d in datasets]
        self._cumsum = np.cumsum([0] + self._lengths)

    def __getitem__(self, idx):
        ds_idx = np.searchsorted(self._cumsum[1:], idx, side="right")
        sample_idx = idx - self._cumsum[ds_idx]
        return self.datasets[ds_idx][sample_idx]

    def __len__(self):
        return int(self._cumsum[-1])


class RandomSplitDataset:
    """Randomly split a dataset into non-overlapping subsets."""

    @staticmethod
    def split(dataset: Dataset, lengths: List[int], seed: Optional[int] = None) -> List[Subset]:
        if seed is not None:
            np.random.seed(seed)
        assert sum(lengths) == len(dataset), "Sum of lengths must equal dataset length"
        indices = np.random.permutation(len(dataset)).tolist()
        result = []
        offset = 0
        for length in lengths:
            result.append(Subset(dataset, indices[offset:offset + length]))
            offset += length
        return result


# ======================================================================
#  Samplers
# ======================================================================

class Sampler:
    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class RandomSampler(Sampler):
    def __init__(self, dataset: Dataset, replacement: bool = False, num_samples: Optional[int] = None):
        self.dataset = dataset
        self.replacement = replacement
        self._num_samples = num_samples or len(dataset)

    def __iter__(self):
        n = len(self.dataset)
        if self.replacement:
            yield from np.random.choice(n, self._num_samples, replace=True).tolist()
        else:
            yield from np.random.permutation(n).tolist()

    def __len__(self):
        return self._num_samples


class WeightedRandomSampler(Sampler):
    def __init__(self, weights: List[float], num_samples: int, replacement: bool = True):
        self.weights = np.array(weights) / np.sum(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        yield from np.random.choice(len(self.weights), self.num_samples,
                                    replace=self.replacement, p=self.weights).tolist()

    def __len__(self):
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        n = len(self.sampler)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


# ======================================================================
#  DataLoader
# ======================================================================

def _default_collate(batch):
    """Collate a list of samples into a batch."""
    if isinstance(batch[0], Tensor):
        return Tensor(np.stack([b.data for b in batch]))
    if isinstance(batch[0], np.ndarray):
        return Tensor(np.stack(batch))
    if isinstance(batch[0], (int, float)):
        return Tensor(np.array(batch))
    if isinstance(batch[0], tuple):
        return tuple(_default_collate(list(s)) for s in zip(*batch))
    if isinstance(batch[0], list):
        return [_default_collate(s) for s in zip(*batch)]
    if isinstance(batch[0], dict):
        return {k: _default_collate([d[k] for d in batch]) for k in batch[0]}
    return batch


class DataLoader:
    """
    Data loader that combines a dataset and a sampler.
    Provides an iterable over the given dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        drop_last: bool = False,
        collate_fn: Optional[Callable] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn or _default_collate
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        if sampler is not None:
            self.sampler = sampler
        elif shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

    def __iter__(self) -> Iterator:
        for indices in self.batch_sampler:
            batch = [self.dataset[i] for i in indices]
            yield self.collate_fn(batch)

    def __len__(self) -> int:
        return len(self.batch_sampler)


# ======================================================================
#  Transforms
# ======================================================================

class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Normalize:
    """Normalize a tensor with mean and std."""

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor((x.data - self.mean) / self.std)


class ToTensor:
    """Convert a numpy array or list to Tensor."""

    def __call__(self, x) -> Tensor:
        if isinstance(x, Tensor):
            return x
        return Tensor(np.array(x, dtype=np.float32))


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if np.random.rand() < self.p:
            return Tensor(np.flip(x.data, axis=-1).copy())
        return x


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if np.random.rand() < self.p:
            return Tensor(np.flip(x.data, axis=-2).copy())
        return x


class RandomCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, x: Tensor) -> Tensor:
        H, W = x.data.shape[-2:]
        th, tw = self.size
        i = np.random.randint(0, H - th + 1)
        j = np.random.randint(0, W - tw + 1)
        return Tensor(x.data[..., i:i+th, j:j+tw])


class CenterCrop:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, x: Tensor) -> Tensor:
        H, W = x.data.shape[-2:]
        th, tw = self.size
        i = (H - th) // 2
        j = (W - tw) // 2
        return Tensor(x.data[..., i:i+th, j:j+tw])


class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, x: Tensor) -> Tensor:
        from scipy.ndimage import zoom
        H, W = x.data.shape[-2:]
        th, tw = self.size
        factors = [1] * (x.ndim - 2) + [th / H, tw / W]
        return Tensor(zoom(x.data, factors))


class GaussianNoise:
    def __init__(self, std: float = 0.1):
        self.std = std

    def __call__(self, x: Tensor) -> Tensor:
        return Tensor(x.data + np.random.randn(*x.data.shape) * self.std)


class RandomErasing:
    def __init__(self, p: float = 0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x: Tensor) -> Tensor:
        if np.random.rand() > self.p:
            return x
        out = x.data.copy()
        H, W = out.shape[-2:]
        area = H * W
        erase_area = np.random.uniform(*self.scale) * area
        ar = np.random.uniform(*self.ratio)
        eh = int(np.sqrt(erase_area * ar))
        ew = int(np.sqrt(erase_area / ar))
        eh, ew = min(eh, H), min(ew, W)
        i = np.random.randint(0, H - eh + 1)
        j = np.random.randint(0, W - ew + 1)
        out[..., i:i+eh, j:j+ew] = self.value
        return Tensor(out)


# ======================================================================
#  Utilities
# ======================================================================

def random_split(dataset: Dataset, lengths: List[int], seed: Optional[int] = None) -> List[Subset]:
    return RandomSplitDataset.split(dataset, lengths, seed)
