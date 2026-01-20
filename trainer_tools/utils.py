import random
from typing import TypeVar
from omegaconf import DictConfig, ListConfig, OmegaConf
from .imports import *

# Selects CUDA if available, otherwise defaults to CPU.
default_device = t.device("cuda" if t.cuda.is_available() else "cpu")


def to_device(x, device=default_device):
    """Recursively moves tensors or collections (lists, tuples, dicts) of tensors to a device."""
    if isinstance(x, t.Tensor):
        return x.to(device)
    if isinstance(x, (tuple, list)):
        return [to_device(el, device) for el in x]
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x


def random_seed(seed: Optional[int], full_determinism: bool = False):
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    if full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def noop(x):
    """returns input"""
    return x


def is_notebook():
    """Check if code is running in a Jupyter notebook."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except (ImportError, AttributeError):
        return False


def flatten_config(cfg, parent_key="", sep="."):
    items = []
    if isinstance(cfg, (DictConfig, ListConfig)):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

T = TypeVar("T")

def init_config_from_args(config_obj: T, config_cls: type[T], kwargs: dict) -> T:
    """Initialize a config object from either a config object or kwargs."""
    if config_obj is None:
        return config_cls(**kwargs)
    elif kwargs:
        config_dict = {k: getattr(config_obj, k) for k in dir(config_obj) if not k.startswith("_")}
        config_dict.update(kwargs)
        return config_cls(**config_dict)
    return config_obj
