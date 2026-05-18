"""
backends.py
-----------
Backend dispatch layer for NumPy / CuPy / PyTorch.

Provides a unified `xp` namespace so that the same numerical code can run on:
  - NumPy (CPU, reference)
  - CuPy (GPU, drop-in NumPy replacement, if available)
  - PyTorch (CPU or GPU, with autograd-compatible tensors)

Each backend exposes the same minimal API:
  - array creation: zeros, ones, linspace, meshgrid, asarray
  - elementary math: exp, log, sqrt, sin, cos, abs, sign, clip, where
  - reductions: sum, mean, max, min, nanmean
  - linear algebra: dot, einsum
  - device transfer: to_device, to_numpy
  - random: random_uniform, random_normal

Usage:
    from backends import get_backend
    xp = get_backend("numpy")        # or "cupy" or "torch"
    x = xp.linspace(0, 1, 100)
    y = xp.sin(x)
    y_np = xp.to_numpy(y)
"""

from __future__ import annotations
import importlib
from typing import Any


class Backend:
    """Unified namespace wrapping NumPy/CuPy/Torch."""

    def __init__(self, name: str):
        self.name = name
        self._setup()

    def _setup(self):
        if self.name == "numpy":
            np = importlib.import_module("numpy")
            self.lib = np
            self.dtype_float = np.float64
            self.dtype_complex = np.complex128
            self.is_torch = False

        elif self.name == "cupy":
            try:
                cp = importlib.import_module("cupy")
            except ImportError as e:
                raise ImportError("CuPy not installed; fall back to numpy.") from e
            self.lib = cp
            self.dtype_float = cp.float64
            self.dtype_complex = cp.complex128
            self.is_torch = False

        elif self.name == "torch":
            torch = importlib.import_module("torch")
            self.lib = torch
            self.dtype_float = torch.float64
            self.dtype_complex = torch.complex128
            self.is_torch = True
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        else:
            raise ValueError(f"Unknown backend: {self.name}")

    # ---- creation -----------------------------------------------------------
    def asarray(self, x, dtype=None):
        if self.is_torch:
            torch = self.lib
            dt = dtype if dtype is not None else self.dtype_float
            if isinstance(x, torch.Tensor):
                return x.to(dtype=dt, device=self._torch_device)
            return torch.tensor(x, dtype=dt, device=self._torch_device)
        return self.lib.asarray(x, dtype=dtype if dtype is not None else self.dtype_float)

    def zeros(self, shape, dtype=None):
        dt = dtype if dtype is not None else self.dtype_float
        if self.is_torch:
            return self.lib.zeros(shape, dtype=dt, device=self._torch_device)
        return self.lib.zeros(shape, dtype=dt)

    def ones(self, shape, dtype=None):
        dt = dtype if dtype is not None else self.dtype_float
        if self.is_torch:
            return self.lib.ones(shape, dtype=dt, device=self._torch_device)
        return self.lib.ones(shape, dtype=dt)

    def linspace(self, start, stop, n):
        if self.is_torch:
            return self.lib.linspace(start, stop, n,
                                     dtype=self.dtype_float,
                                     device=self._torch_device)
        return self.lib.linspace(start, stop, n, dtype=self.dtype_float)

    def meshgrid(self, *xs, indexing="xy"):
        if self.is_torch:
            return self.lib.meshgrid(*xs, indexing=indexing)
        return self.lib.meshgrid(*xs, indexing=indexing)

    # ---- math ---------------------------------------------------------------
    def __getattr__(self, name: str):
        """Delegate math functions (exp, log, sqrt, etc.) to the underlying library."""
        # Avoid infinite recursion for internal attributes
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.lib, name)

    # ---- where / clip -------------------------------------------------------
    def where(self, cond, x, y):
        return self.lib.where(cond, x, y)

    def clip(self, x, lo, hi):
        if self.is_torch:
            return self.lib.clamp(x, lo, hi)
        return self.lib.clip(x, lo, hi)

    # ---- reductions ---------------------------------------------------------
    def sum(self, x, axis=None):
        if axis is None:
            return self.lib.sum(x)
        return self.lib.sum(x, axis=axis) if not self.is_torch \
            else self.lib.sum(x, dim=axis)

    def mean(self, x, axis=None):
        if axis is None:
            return self.lib.mean(x)
        return self.lib.mean(x, axis=axis) if not self.is_torch \
            else self.lib.mean(x, dim=axis)

    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        """Trapezoidal integration. Wraps numpy/cupy/torch differently."""
        if self.is_torch:
            # torch.trapezoid signature: (y, x=None, *, dx=None, dim=-1)
            # but dx and x are mutually exclusive
            if x is not None:
                return self.lib.trapezoid(y, x=x, dim=axis)
            return self.lib.trapezoid(y, dx=dx, dim=axis)
        # numpy >= 2.0 renamed trapz -> trapezoid
        if hasattr(self.lib, "trapezoid"):
            return self.lib.trapezoid(y, x=x, dx=dx, axis=axis)
        return self.lib.trapz(y, x=x, dx=dx, axis=axis)

    # ---- random -------------------------------------------------------------
    def random_uniform(self, lo, hi, size, seed=None):
        if self.is_torch:
            torch = self.lib
            gen = torch.Generator(device=self._torch_device)
            if seed is not None:
                gen.manual_seed(int(seed))
            t = torch.rand(size, generator=gen,
                           dtype=self.dtype_float, device=self._torch_device)
            return lo + (hi - lo) * t
        rng = self.lib.random.default_rng(seed)
        return rng.uniform(lo, hi, size)

    def random_normal(self, mean, std, size, seed=None):
        if self.is_torch:
            torch = self.lib
            gen = torch.Generator(device=self._torch_device)
            if seed is not None:
                gen.manual_seed(int(seed))
            t = torch.randn(size, generator=gen,
                            dtype=self.dtype_float, device=self._torch_device)
            return mean + std * t
        rng = self.lib.random.default_rng(seed)
        return rng.normal(mean, std, size)

    # ---- device / numpy conversion ------------------------------------------
    def to_numpy(self, x) -> Any:
        if self.is_torch:
            return x.detach().cpu().numpy()
        if self.name == "cupy":
            return x.get()
        return x

    def device_info(self) -> str:
        if self.is_torch:
            return f"torch[{self._torch_device}]"
        if self.name == "cupy":
            return "cupy[gpu]"
        return "numpy[cpu]"


def get_backend(name: str) -> Backend:
    """Factory. Falls back to numpy if requested backend unavailable."""
    try:
        return Backend(name)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[backend] '{name}' unavailable ({e}); falling back to numpy.")
        return Backend("numpy")


def available_backends() -> list[str]:
    """Return list of installed backends."""
    out = ["numpy"]
    for name in ("cupy", "torch"):
        try:
            importlib.import_module(name)
            out.append(name)
        except ImportError:
            pass
    return out


if __name__ == "__main__":
    # Self-test
    for name in available_backends():
        xp = get_backend(name)
        x = xp.linspace(0.0, 1.0, 5)
        y = xp.sin(x)
        print(f"{xp.device_info():20s} sin(linspace(0,1,5)) = {xp.to_numpy(y)}")
