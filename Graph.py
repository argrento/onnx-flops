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

from collections import OrderedDict
from typing import List

from onnx.onnx_ml_pb2 import GraphProto, NodeProto, ValueInfoProto
from rich.console import Console
from rich.table import Table

from Operator import Operator
from Tensor import Tensor


class Graph:
    def __init__(self, graph: GraphProto):
        self.graph = graph
        self.name: str = graph.name

        self.tensors: OrderedDict[str, Tensor] = OrderedDict()
        self._parse_tensors()
        self.tensors["input"] = self._parse_io(self.graph.input[0], "input")
        self.tensors["output"] = self._parse_io(self.graph.output[0], "output")

        self.operators: List[Operator] = []
        self._parse_ops()
        self._create_op_outputs()

    def _parse_tensors(self):
        """
        Parse tensors for operators
        :return: None
        """
        for tensor_proto in self.graph.initializer:
            t = Tensor(proto=tensor_proto)
            self.tensors[t.name] = t

    def _parse_io(self, proto: ValueInfoProto, name: str):
        """
        Parse Input and Output tensors
        :return: None
        """
        tensor_shape = []
        for dim in proto.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                # Dimension value is predefined
                tensor_shape.append(dim.dim_value)
            else:
                # Dimension value is unknown
                tensor_shape.append(1)
        tensor_type = proto.type.tensor_type.elem_type
        return Tensor(dims=tensor_shape, data_type=tensor_type, name=name)

    def _parse_ops(self):
        for op_proto in self.graph.node:
            op = Operator(op_proto)
            if op.op_type == "Conv":
                if len(self.tensors[op.inputs[1]].dims) == 4:
                    # detect weight and bias
                    op.weight = self.tensors[op.inputs[1]]
                    op.bias = self.tensors[op.inputs[2]]
                else:
                    op.weight = self.tensors[op.inputs[2]]
                    op.bias = self.tensors[op.inputs[1]]
            self.operators.append(op)

    def _create_op_outputs(self):
        input_data = self.tensors['input']
        data_type = self.tensors['input'].data_type
        for op in self.operators:
            output_dims = op.compute_output_shape(input_data.dims)
            output_name = op.outputs[0]
            t = Tensor(dims=output_dims, data_type=data_type, name=output_name)
            self.tensors[t.name] = t
            input_data = t

    def summary(self):
        console = Console()

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Operator", style="dim")
        table.add_column("FLOP")
        table.add_column("Bytes IN")
        table.add_column("Bytes OUT")

        for op in self.operators:
            op_name = op.name
            op_in_bytes = 0
            # Find inputs
            for in_name in op.inputs:
                if in_name in self.tensors:
                    op_in_bytes += self.tensors[in_name].size

            op_out_bytes = 0
            # Find inputs
            for out_name in op.outputs:
                if out_name in self.tensors:
                    op_out_bytes += self.tensors[out_name].size

            table.add_row(op.name, str(op.compute_flop()), str(op_in_bytes), str(op_out_bytes))

        console.print(table)
