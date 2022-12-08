from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any
dtypedef = Any

class RotNetBlock(nn.Module):
    cnn_channels: int
    norm: ModuleDef
    dtype: dtypedef
    kernel_init: Callable

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.cnn_channels, kernel_size=(3, 3), dtype=self.dtype, kernel_init=self.kernel_init)(x)
        x = self.norm()(x)
        x = nn.relu(x)
        return x
    
class Features(nn.Module):
    cnn_channels: int
    num_blocks: int
    dtype: dtypedef
    kernel_init: Callable
    
    @nn.compact
    def __call__(self, x, train):
        norm = partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        for _ in range(self.num_blocks):
            x = RotNetBlock(cnn_channels=self.cnn_channels, norm=norm, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class Classifier(nn.Module):
    cnn_channels: int
    num_blocks: int
    num_classes: int
    dtype: dtypedef
    kernel_init: Callable
    
    @nn.compact
    def __call__(self, x, train):
        norm = partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        for _ in range(self.num_blocks):
            x = RotNetBlock(cnn_channels=self.cnn_channels, norm=norm, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class RotNet(nn.Module):
    cnn_channels: int
    num_blocks_features: int
    num_blocks_classifier: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    # ---------------------- Refactor Module into Submodules --------------------- #
    # https://flax.readthedocs.io/en/latest/guides/extracting_intermediates.html#refactor-module-into-submodules
    def setup(self):
        self.features = Features(self.cnn_channels, self.num_blocks_features, self.dtype, self.kernel_init)
        self.classifier = Classifier(self.cnn_channels, self.num_blocks_classifier, self.num_classes, self.dtype, self.kernel_init)

    def __call__(self, x, train):
        x = self.features(x, train)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x, train)
        return x

def rotnet_constructor(model_arch):
    cnn_channels = 64
    if model_arch == 'rotnet3_feat3':
        num_blocks, num_blocks_features = 3, 3
    else:
        raise ValueError()
    
    return RotNet(cnn_channels, num_blocks_features, num_blocks - num_blocks_features, 4)
