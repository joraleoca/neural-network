from typing import Callable

from .tensor import Tensor

def ensure_input_tensor(func: Callable) -> Callable:
    """Decorator to ensure the input is a Tensor."""
    def wrapper(input, *args, **kwargs) -> Callable:
        if not isinstance(input, Tensor):
            input = Tensor(input)
        return func(input, *args, **kwargs)
    return wrapper
