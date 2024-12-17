import math
from typing import Union, Tuple, Optional
import torch
from torch.nn import functional as nnf


def calc_same_pad(i: int, k: int, s: int, d: int) -> int:
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)


class MaxPool2dSamePadding(torch.nn.MaxPool2d):

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = calc_same_pad(
            i=ih,
            k=self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0],
            s=self.stride if isinstance(self.stride, int) else self.stride[0],
            d=self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        )
        pad_w = calc_same_pad(
            i=iw,
            k=self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0],
            s=self.stride if isinstance(self.stride, int) else self.stride[0],
            d=self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        )

        if pad_h > 0 or pad_w > 0:
            x = nnf.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return nnf.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices
        )


class Conv2dSame(torch.nn.Conv2d):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = calc_same_pad(
            i=ih,
            k=self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0],
            s=self.stride if isinstance(self.stride, int) else self.stride[0],
            d=self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        )
        pad_w = calc_same_pad(
            i=iw,
            k=self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0],
            s=self.stride if isinstance(self.stride, int) else self.stride[0],
            d=self.dilation if isinstance(self.dilation, int) else self.dilation[0]
        )

        if pad_h > 0 or pad_w > 0:
            x = nnf.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return nnf.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
