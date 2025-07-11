from typing import Literal
import torch
import platform


def check_sys_arch() -> Literal["cpu", "mps", "cuda"]:
    is_metal = (
        platform.machine() == "arm64"
        and "mac" in platform.platform().lower()
        and torch.backends.mps.is_available()
    )

    if is_metal:
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device
