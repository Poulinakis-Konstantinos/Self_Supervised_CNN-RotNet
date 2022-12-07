import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
# from utils_flax import compute_weight_decay
import numpy as np

ModuleDef = Any
dtypedef = Any

# class Sequential(nn.Module):
#   layers: Sequence[nn.Module]

#   def __call__(self, x, train):
#     for layer in self.layers:
#       x = layer(x, train)
#     return x

#   def append(self, layer):
#     self.list_layers = list(self.layers)
#     self.list_layers.append(layer)
#     self.layers = Sequence(self.list_layers)
      

class PredNetBlock(nn.Module):
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

class Classifier(nn.Module):
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)(x)
        return x

class PredNet(nn.Module):
    backbone: nn.Module
    cnn_channels: int
    num_blocks: int
    num_classes: int
    dtype: dtypedef = jnp.float32
    kernel_init: Callable = nn.initializers.glorot_uniform()

    def setup(self):
        self.classifier = Classifier(num_classes=self.num_classes, dtype=self.dtype, kernel_init=self.kernel_init)

    def __call__(self, x, train):
        x = self.backbone(x, train)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

def _prednet(backbone, cnn_channels, num_blocks, num_classes):
    model = PredNet(backbone, cnn_channels=cnn_channels, num_blocks=num_blocks, num_classes=num_classes)
    return model

def PredNet3(backbone):
    return _prednet(backbone, cnn_channels=64, num_blocks=3, num_classes=10)


def PredNet4(backbone):
    return _prednet(backbone, cnn_channels=64, num_blocks=4, num_classes=10)


def PredNet5(backbone):
    return _prednet(backbone, cnn_channels=64, num_blocks=5, num_classes=10)
