import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)


class QATLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.activation_fake_quant = FakeQuantize(
            observer=MovingAverageMinMaxObserver, dtype=torch.qint8,
            qscheme=torch.per_tensor_affine, quant_min=-127, quant_max=127)
        self.weight_fake_quant = FakeQuantize(
            observer=MovingAveragePerChannelMinMaxObserver, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric, ch_axis=0, quant_min=-127, quant_max=127)

    def forward(self, value):
        if self.training:
            value = self.activation_fake_quant(value)
            weight = self.weight_fake_quant(self.weight)
        else:
            activation = self.activation_fake_quant
            weight_quant = self.weight_fake_quant
            if activation.fake_quant_enabled.item():
                value = torch.fake_quantize_per_tensor_affine(
                    value, activation.scale, activation.zero_point, -127, 127)
            weight = self.weight
            if weight_quant.fake_quant_enabled.item():
                weight = torch.fake_quantize_per_channel_affine(
                    weight, weight_quant.scale, weight_quant.zero_point, 0, -127, 127)
        return F.linear(value, weight, self.bias)


def replace_linear(sequence):
    for index, module in enumerate(sequence):
        if isinstance(module, nn.Linear):
            qat = QATLinear(module.in_features, module.out_features, module.bias is not None)
            qat.weight = module.weight
            qat.bias = module.bias
            sequence[index] = qat
