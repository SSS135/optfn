from .auto_norm import AutoNorm
from ppo_pytorch.actors.norm_factory import NormFactory


class AutoNormFactory(NormFactory):
    def __init__(self, group_size=16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_size = group_size

    def _fc_factory(self, num_features):
        return AutoNorm(num_features, self.group_size)

    _cnn_factory = _fc_factory