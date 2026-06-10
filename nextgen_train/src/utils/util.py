"""Module with utility functions."""

import hashlib
import subprocess

import torch
from omegaconf import DictConfig, ListConfig


def torch_worker_init_fn(base_seed: int):
    """Set random seed for this worker (different workers get different random seeds)"""

    def init_fn(worker_id: int):
        torch.manual_seed(base_seed + worker_id)

    return init_fn


def torch_float16() -> torch.dtype:
    """Utility function that wraps torch.float16 dtype in a callable.

    Useful for specifying in Hydra configs with `_target_`.
    """
    return torch.float16


def torch_bfloat16() -> torch.dtype:
    """Utility function that wraps torch.float16 dtype in a callable.

    Useful for specifying in Hydra configs with `_target_`.
    """
    return torch.bfloat16


def flatten_dict(nested_dict: dict | DictConfig, sep: str = "/", key_prefix: str | None = None):
    """Flattens a potentially nested dictionary.

    Args:
      nested_dict: The input dictionary.
      key_prefix: The prefix for the keys of the flattened dictionary.
      sep: Separator for flattening the dict.

    Returns:
      A flattened dictionary.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = key_prefix + sep + k if key_prefix else k
        if isinstance(v, dict | DictConfig):
            items.extend(flatten_dict(v, key_prefix=new_key, sep=sep).items())
        elif isinstance(v, list | ListConfig):
            for i, item in enumerate(v):
                new_key_prefix = new_key + sep + str(i)
                if isinstance(item, dict | DictConfig):
                    items.extend(flatten_dict(item, key_prefix=new_key_prefix, sep=sep).items())
                else:
                    items.append((new_key_prefix, item))
        else:
            items.append((new_key, v))
    return dict(items)


def get_git_rev_hash() -> str | None:
    """Get the current git revision hash."""
    try:
        rev_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8").strip()  # noqa: S607,S603
        return rev_hash
    except subprocess.CalledProcessError:
        return None


def get_working_tree_hash() -> str | None:
    """Calculates the SHA-256 hash of the Git working tree.

    Returns:
        str: The SHA-256 hash of the working tree.
    """

    # Get the output of `git ls-tree -r HEAD`
    try:
        tree_output = subprocess.check_output(  # noqa: S603
            ["git", "ls-tree", "-r", "HEAD"],  # noqa: S607
            stderr=subprocess.DEVNULL,  # Suppress potential errors
            universal_newlines=True,  # Get output as a string
        )
    except subprocess.CalledProcessError:
        return None  # Handle cases where Git command fails

    # Calculate the SHA-256 hash of the output
    hash_object = hashlib.sha256(tree_output.encode())
    return hash_object.hexdigest()
