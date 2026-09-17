from scaffold_codec.train import main

from .config import load_config
from .model import CompressedGaussianModel


if __name__ == "__main__":
    main(CompressedGaussianModel, load_config)
