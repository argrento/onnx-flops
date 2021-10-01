# MIT License
#
# Copyright (c) 2021 Kirill Snezhko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Optional

import numpy as np
from onnx.onnx_ml_pb2 import NodeProto

from Tensor import Tensor


class Operator:
    def __init__(self, proto: NodeProto):
        self.proto: NodeProto = proto
        self.name: str = proto.name
        self.op_type: str = proto.op_type
        self.inputs: List[str] = proto.input
        self.outputs: List[str] = proto.output
        self.input_shape: List[int] = None
        self.output_shape: List[int] = None

        # Optional convolutional parameters
        self.dilations: Optional[List[int]] = None
        self.group: Optional[int] = None
        self.kernel_shape: Optional[List[int]] = None
        self.pads: Optional[List[int]] = None
        self.strides: Optional[List[int]] = None

        self.weight: Optional[Tensor] = None
        self.bias: Optional[Tensor] = None

    def compute_output_shape(self, input_shape: List[List[int]]):
        self.input_shape = [x for x in input_shape]
        self.output_shape = [x for x in self.input_shape]
        if self.op_type == "Conv":
            for attr in self.proto.attribute:
                if attr.ints:
                    # Attribute value is a list of ints
                    attr_val = attr.ints
                else:
                    attr_val = attr.i
                # Set attribute value
                self.__dict__[attr.name] = attr_val
            # Equations from
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            self.output_shape[1] = self.weight.dims[0]
            self.output_shape[2] = (self.input_shape[2] +
                                    2 * self.pads[0] -
                                    self.dilations[0] * (self.kernel_shape[0] - 1) - 1) // \
                                   self.strides[0] + 1
            self.output_shape[3] = (self.input_shape[3] +
                                    2 * self.pads[1] -
                                    self.dilations[1] * (self.kernel_shape[1] - 1) - 1) // \
                                   self.strides[1] + 1
        elif self.op_type in ["Clip"]:
            self.output_shape = [x for x in self.input_shape]
        return self.output_shape

    def compute_flop(self):
        # Eq. 2, p. 2, https://arxiv.org/pdf/1704.04861.pdf
        flop = 0
        if self.op_type == "Conv":
            flop = self.input_shape[1] * self.weight.dims[0] * self.kernel_shape[0] * \
                   self.kernel_shape[1] * self.input_shape[2] * self.input_shape[3]
        elif self.op_type in ["Clip", "Add", "GlobalAveragePool"]:
            flop = np.prod(self.input_shape)
        elif self.op_type == "Gemm":
            flop = np.prod(self.input_shape) * np.prod(self.output_shape)

        return flop