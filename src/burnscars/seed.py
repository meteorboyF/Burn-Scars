"""One seeding utility, called exactly once per entrypoint."""
import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> dict:
    """Seed python, numpy and (if installed) torch + CUDA.

    Returns a dict describing what was seeded, for the run manifest. Full
    determinism is not available for every CUDA kernel; where torch refuses a
    deterministic op we fall back to warn-only and record that fact.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    info = {"seed": seed, "python": True, "numpy": True, "torch": False,
            "torch_deterministic": None}
    try:
        import torch  # noqa: PLC0415  (optional at this phase)
    except ImportError:
        return info
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    info["torch"] = True
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        info["torch_deterministic"] = "warn_only"
    return info
