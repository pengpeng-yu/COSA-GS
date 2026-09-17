import math

import torch
import torch.nn as nn

from .int_nn_ops import ONE, WeightRange, IntLinear, IntRequant


def round_divide(value, divisor):
    """Signed nearest rounding, half ties away from zero; positive divisor."""
    assert value.dtype == torch.int64
    assert type(divisor) is int or (isinstance(divisor, torch.Tensor) and divisor.dtype == torch.int64)
    magnitude = value.abs()
    quotient = torch.div(magnitude, divisor, rounding_mode="floor")
    remainder = magnitude.remainder(divisor)
    rounded = quotient + (remainder >= (divisor + 1) // 2).to(torch.int64)
    return torch.where(value < 0, -rounded, rounded)


def checked_int32(value):
    assert value.dtype in (torch.int64, torch.float64)
    assert ((value >= -(1 << 31)) & (value < (1 << 31))).all()
    return value.to(torch.int32)


def integer_multiplier(ratio):
    assert ratio.dtype == torch.float64
    ratio = ratio.cpu().reshape(-1)
    assert torch.isfinite(ratio).all() and (ratio > 0).all()
    requant_shift = min(62, math.floor(math.log2(((1 << 31) - 1) / ratio.max().item())))
    assert requant_shift >= 0, "Requant ratio is too large"
    requant_mul = (ratio * (1 << requant_shift)).round().to(torch.int64)
    assert ((requant_mul > 0) & (requant_mul < (1 << 31))).all()
    return requant_mul.to(torch.uint32), requant_shift


@torch.no_grad()
def export_sequential(sequence, import_parameters=True):
    """Each Linear(+GELU) becomes one GEMM and one fused pointwise operation."""
    result = [IntRequant()]
    if import_parameters:
        activation = sequence[0].activation_fake_quant
        scale_out = activation.scale.detach().cpu().double()
        requant_mul, requant_shift = integer_multiplier(1.0 / (ONE * scale_out))
        result[0].requant_mul.copy_(requant_mul)
        result[0].requant_shift.fill_(requant_shift)
        result[0].zero_point_out.copy_(activation.zero_point)

    index = 0
    while index < len(sequence):
        linear = sequence[index]
        assert isinstance(linear, nn.Linear)
        with_gelu = index + 1 < len(sequence) and isinstance(sequence[index + 1], nn.GELU)
        next_index = index + 1 + int(with_gelu)
        next_linear = sequence[next_index] if next_index < len(sequence) else None
        if not with_gelu and next_linear is not None:
            raise NotImplementedError("Consecutive Linear layers without GELU are not supported.")
        module = IntLinear(linear.in_features, linear.out_features, with_gelu, next_linear is not None)

        if import_parameters:
            scale_in = linear.activation_fake_quant.scale.detach().cpu().double()
            zero_point_in = linear.activation_fake_quant.zero_point.detach().cpu().to(torch.int64)
            scale_weight = linear.weight_fake_quant.scale.detach().cpu().double()
            assert scale_in.numel() == 1 and scale_weight.numel() == linear.out_features

            weight = torch.quantize_per_channel(
                linear.weight.detach().cpu(), scale_weight,
                linear.weight_fake_quant.zero_point.detach().cpu().long(),
                0, torch.qint8).int_repr().clamp(-WeightRange, WeightRange)

            bias = -zero_point_in * weight.to(torch.int64).sum(1)
            if linear.bias is not None:
                bias = bias.to(torch.float64) + (linear.bias.detach().cpu().double() / (scale_in * scale_weight)).round()
            module.weight.copy_(weight)
            module.bias.copy_(checked_int32(bias))

            requant_mul, requant_shift = integer_multiplier(scale_in * scale_weight * ONE)
            module.requant_mul.copy_(requant_mul)
            module.requant_shift.fill_(requant_shift)

            if module.out_scaled_int:
                assert next_linear is not None
                scale_out = next_linear.activation_fake_quant.scale.detach().cpu().double()
                zero_point_out = next_linear.activation_fake_quant.zero_point.detach().cpu()
                requant_mul, requant_shift = integer_multiplier(1.0 / (ONE * scale_out))
                module.requant_mul_out.copy_(requant_mul)
                module.requant_shift_out.fill_(requant_shift)
                module.zero_point_out.copy_(zero_point_out)

        result.append(module)
        index = next_index

    return nn.Sequential(*result)
