"""
To calibrate a trained floating-point checkpoint for INT8 inference and evaluate it
without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scaffold_codec_int.convert_and_evaluate \
  --config config/codec30k_mipnerf360.yaml \
  --source-path datasets/mipnerf360/bicycle \
  --checkpoint outputs/mipnerf360_30k/bicycle_lmb0.0006/checkpoints/final.pt \
  --output outputs/mipnerf360_30k_ptq/bicycle_lmb0.0006
```

Calibration uses all anchors in the checkpoint. The converted checkpoint is saved
to `checkpoints/ptq.pt` under the output directory, followed by integer compression,
decompression, and evaluation with the same options as checkpoint evaluation.
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from scaffold_codec.compress_and_evaluate import backup_existing_output, parse_args
from scaffold_codec.model import CompressedGaussianModel as FloatModel

from .compress_and_evaluate import main as evaluate_main
from .config import load_config
from .model import CompressedGaussianModel
from .qat import QATLinear


class PTQLinear(QATLinear):
    def forward(self, value):
        value = self.activation_fake_quant(value)
        weight = self.weight_fake_quant(self.weight)
        return F.linear(value, weight, self.bias)


@torch.no_grad()
def convert_checkpoint(checkpoint, model_config):
    model = CompressedGaussianModel(model_config).cuda()
    FloatModel.restore_model_state(model, checkpoint)
    quantizers = []
    for name, module in list(model.named_modules()):
        if isinstance(module, QATLinear):
            linear = PTQLinear(module.in_features, module.out_features, module.bias is not None).to(module.weight.device)
            linear.weight = module.weight
            linear.bias = module.bias
            model.set_submodule(name, linear)
            quantizers.extend((linear.activation_fake_quant, linear.weight_fake_quant))

    model.eval()
    for quantizer in quantizers:
        quantizer.disable_fake_quant()
        quantizer.enable_observer()
    # Calibrate on all anchors with deterministic evaluation-time reconstructions.
    model(rd_iteration=0)
    for quantizer in quantizers:
        quantizer.disable_observer()
        quantizer.enable_fake_quant()
    return model.checkpoint_dict(checkpoint["iteration"])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    converted = convert_checkpoint(checkpoint, cfg.model)
    path = Path(args.output) / "checkpoints" / "ptq.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        backup_existing_output(path)
    torch.save(converted, path)
    del checkpoint, converted
    print(f"Saved PTQ checkpoint: {path}", flush=True)
    args.checkpoint = str(path)
    evaluate_main(CompressedGaussianModel, load_config, args=args)


if __name__ == "__main__":
    main()
