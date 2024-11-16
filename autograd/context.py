from typing import Callable

import tensor


class Context:
    backwards_func: Callable

    data: tuple["tensor.Tensor", ...]
