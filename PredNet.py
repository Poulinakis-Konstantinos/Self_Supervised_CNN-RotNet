from functools import partial
from typing import Any, Callable

import jax.numpy as jnp
import flax.linen as nn

ModuleDef = Any
dtypedef = Any

class PredNetBlock(nn.Module):
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
            x = PredNetBlock(cnn_channels=self.cnn_channels, norm=norm, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class PredNet(nn.Module):
    backbone: nn.Module
    cnn_channels: int
    num_blocks_classifier: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    def setup(self):
        self.classifier = Classifier(self.cnn_channels, self.num_blocks_classifier, self.num_classes, self.dtype, self.kernel_init)

    def __call__(self, x, train):
        x = self.backbone(x, train)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x, train)
        return x

def prednet_constructor(model_arch, backbone):
    cnn_channels = 64
    if model_arch == 'prednet3':
        num_blocks_classifier = 3
    else:
        raise ValueError()
    
    return PredNet(backbone, cnn_channels, num_blocks_classifier, 10)
