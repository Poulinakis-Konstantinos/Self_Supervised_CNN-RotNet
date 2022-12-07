import jax
import tensorflow as tf	
from typing import Any, Callable, Sequence, Optional, List
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
from utils_flax import compute_weight_decay
import numpy as np

from jax.nn import Seqential, relu, softmax
from flax.training import train_state, checkpoints

import optax

from RotNet import RotNet3


SAVE_PATH = 'Saved_models'

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


class Head(nn.Module):
    dense_layers: List[int]
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)
    num_classes: int
    
    @nn.compact
    def __call__(self, inputs, train: bool):
        # if shape (32, 32, 3) : need flatten layer
        # if shape(1, 32, 32, 3): not need flatten ?
        x = inputs.reshape(inputs.shape[0], -1)
        for layers in self.dense_layers:
            x = nn.Dense(layers)(x)
            x = nn.relu(x)
            x = self.batch_norm_cls(use_running_average=not train)(x)
        x = nn.Dense(self.num_classes)(x)
        x = nn.softmax(x)
        return x


class TransferModel(nn.Module):
    backbone: Seqential
    head: Head
    cnn_layers: List[int]
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)
    
    @nn.compact
    def __call__(self, input, train: bool):
        x = input
        if self.backbone is not None:
            x = self.backbone(x)
        else:
            for layers in self.cnn_layers:
                x = nn.Conv(features=layers, kernel_size=(3, 3), dtype=self.dtype, kernel_init=self.kernel_init)
                x = nn.relu(x)
                x = self.batch_norm_cls(use_running_average=not train)(x)
        x = self.head(x)
        return x








def PredNet(cnn_layers, dense_layers, in_shape=(1,32,32,3), classes=10, transfer=True, base_model_name='rotnet_v1', lr = 0.1, momentum = 0.01):
    
    model = TransferModel()
    
    
    

    return model


def PredNet_constructor(build_instructions: dict):
    
    model = None
    
    return model
    
    