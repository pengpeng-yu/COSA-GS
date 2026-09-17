from .config import ExperimentConfig, load_config


def __getattr__(name):
    if name == "CompressedGaussianModel":
        from .model import CompressedGaussianModel
        return CompressedGaussianModel
    raise AttributeError(name)
