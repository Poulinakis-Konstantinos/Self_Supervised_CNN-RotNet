import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
from utils_flax import compute_weight_decay
import numpy as np

ModuleDef = Any
dtypedef = Any

class RotNetBlock(nn.Module):
    cnn_channels: int
    norm: ModuleDef
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.cnn_channels, kernel_size=(3, 3), dtype=self.dtype, kernel_init=self.kernel_init)(x)
        x = self.norm()(x)
        x = nn.relu(x)
        return x

class RotNet(nn.Module):
    cnn_channels: int
    num_blocks: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    @nn.compact
    def __call__(self, x, train):
        norm = partial(nn.BatchNorm, use_running_average=not train, dtype=self.dtype)
        for _ in range(self.num_blocks):
            x = RotNetBlock(cnn_channels=self.cnn_channels, norm=norm)(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

def _rotnet(cnn_channels, num_blocks, num_classes):
    model = RotNet(cnn_channels=cnn_channels, num_blocks=num_blocks, num_classes=num_classes)
    return model

def RotNet3():
    return _rotnet(cnn_channels=64, num_blocks=3, num_classes=4)


def RotNet4():
    return _rotnet(cnn_channels=64, num_blocks=4, num_classes=4)


def RotNet5():
    return _rotnet(cnn_channels=64, num_blocks=5, num_classes=4)
