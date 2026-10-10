from ..config import Config
from .toolkit import Toolkit


def get_toolkit(cfg: Config) -> Toolkit:
    if cfg.offline_demo:
        from .demo import DemoToolkit

        return DemoToolkit(cfg)
    return Toolkit(cfg)
