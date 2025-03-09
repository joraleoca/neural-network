import cupy as cp

from ..function import Function
from ... import tensor


class As_Strided(Function):
    """Creates a view of a tensor with the specified strides and shape."""
    
    __slots__ = "shape", "strides"

    def __init__(self, a: "tensor.Tensor", *, shape: tuple[int], strides: tuple[int]) -> None:
        self.args = (a,)
        self.shape = shape
        self.strides = strides

    def __call__(self, *, inplace: bool = False) -> "tensor.Tensor":
        if inplace:
            raise NotImplementedError("Inplace as_strided is not supported. It always returns a view.")

        a = self.args[0]

        xp = cp.get_array_module(a.data)

        return self._create_output_tensor(xp.lib.stride_tricks.as_strided(a.data, self.shape, self.strides))

    def backward(self) -> None:
        a = self.args[0]
        
        if not a.requires_grad:
            return
            
        grad = self.result.grad
        
        xp = cp.get_array_module(a.data)
        gr = xp.zeros_like(a.data)

        strides = xp.array(self.strides) // self.result.data.itemsize
        indices = xp.indices(self.shape).reshape(len(self.shape), -1)
        offsets = xp.tensordot(strides, indices, axes=1).astype(int)
    
        xp.add.at(gr.ravel(), offsets, grad.ravel())

        if a.grad is None:
            a.grad = gr
        else:
            a.grad += gr
        
        