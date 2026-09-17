from dataclasses import dataclass, field

from scaffold_codec.config import ExperimentConfig as BaseExperimentConfig
from scaffold_codec.config import ModelConfig as BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    qat_start_iteration: int = 0
    qat_observer_freeze_iteration: int = 25_000


@dataclass
class ExperimentConfig(BaseExperimentConfig):
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path, overrides=()):
    cfg = ExperimentConfig()
    cfg.merge_with_yaml(path)
    cfg.merge_with_dotlist(overrides)
    cfg.check()
    return cfg
