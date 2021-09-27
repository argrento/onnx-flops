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

from onnx.onnx_ml_pb2 import TensorProto
from functools import reduce

from typing import Optional, List


class Tensor:
    def __init__(self, proto: Optional[TensorProto] = None, dims: Optional[List[int]] = None,
                 data_type: Optional[int] = None, name: Optional[str] = None):
        if proto:
            self.dims = [x for x in proto.dims]
            self.data_type = proto.data_type
            self.name = proto.name
        else:
            self.dims = dims
            self.data_type = data_type
            self.name = name
        if len(self.dims) > 1:
            self.size = reduce(lambda x, y: x * y, self.dims) * self.data_type
        else:
            self.size = self.dims[0] * self.data_type

    def __str__(self):
        return f"Tensor '{self.name}', shape: {self.dims}, size: {self.size} bytes."

    def __repr__(self):
        return f"'Tensor '{self.name}', shape: {self.dims}'"
