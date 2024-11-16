from tensor import Tensor
from autograd.operations import Exp


def exp(input: Tensor, *, inplace: bool = False) -> Tensor:
    return input.apply_operation(inplace=inplace, operation=Exp)
