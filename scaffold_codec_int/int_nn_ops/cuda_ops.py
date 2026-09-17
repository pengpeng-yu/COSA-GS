import torch
import torch.nn as nn

from .build import int_nn_ops_ext


SharedFxpShift = 20
ONE = 1 << SharedFxpShift
WeightRange = (1 << 7) - 1
ActRange = (1 << 7) - 1


def gelu(input):
    return int_nn_ops_ext.gelu(input.contiguous())


class IntRequant(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("requant_mul", torch.ones(1, dtype=torch.uint32))
        self.register_buffer("requant_shift", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("zero_point_out", torch.zeros(1, dtype=torch.int32))

    def forward(self, input):
        return int_nn_ops_ext.requant_to_int8(
            input.contiguous(), self.requant_mul, self.requant_shift, self.zero_point_out)


class IntLinear(nn.Module):
    def __init__(self, in_ch, out_ch, with_gelu, out_scaled_int):
        super().__init__()
        self.with_gelu = with_gelu
        self.out_scaled_int = out_scaled_int
        self.register_buffer("weight", torch.empty(out_ch, in_ch, dtype=torch.int8))
        self.register_buffer("bias", torch.empty(out_ch, dtype=torch.int32))
        self.register_buffer("requant_mul", torch.ones(out_ch, dtype=torch.uint32))
        self.register_buffer("requant_shift", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("requant_mul_out", torch.ones(1, dtype=torch.uint32))
        self.register_buffer("requant_shift_out", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("zero_point_out", torch.zeros(1, dtype=torch.int32))

    def forward(self, input):
        assert input.dtype == torch.int8
        mm_out = torch.empty((input.shape[0], self.weight.shape[0]), device=input.device, dtype=torch.int32)
        int_nn_ops_ext.cutlass_gemm_int8(input.contiguous(), self.weight, self.bias, mm_out)
        if self.with_gelu:
            if self.out_scaled_int:
                return int_nn_ops_ext.requant_gelu_requant_to_int8(
                    mm_out, self.requant_mul, self.requant_shift,
                    self.requant_mul_out, self.requant_shift_out, self.zero_point_out)
            return int_nn_ops_ext.requant_gelu_to_int32(
                mm_out, self.requant_mul, self.requant_shift)
        return int_nn_ops_ext.requant_to_int32(
            mm_out, self.requant_mul, self.requant_shift, self.zero_point_out)
