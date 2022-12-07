import jax
import tensorflow as tf
from typing import Any, Callable, Sequence, Optional, List
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
import copy
from functools import partial
import numpy as np
from flax.training import train_state, checkpoints

import optax

from RotNet import RotNet3


SAVE_PATH = "Saved_models"

# Generate key which is used to generate random numbers
key = random.PRNGKey(1)


class Head(nn.Module):
    dense_layers: List[int]
    num_classes: int
    cnn_layers: List[int]
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)

    @nn.compact
    def __call__(self, inputs, train: bool):
        for layers in self.cnn_layers:
            x = nn.Conv(
                features=layers,
                kernel_size=(3, 3),
                dtype=self.dtype,
                kernel_init=self.kernel_init,
            )
            x = nn.relu(x)
            x = self.batch_norm_cls(use_running_average=not train)(x)
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
    backbone: nn.Module
    head: nn.Module
    cnn_layers: List[int]
    batch_norm_cls: partial = partial(nn.BatchNorm, momentum=0.9)

    @nn.compact
    def __call__(self, input, train: bool):
        x = input
        if self.backbone is not None:
            x = self.backbone(x)

        else:
            for layers in self.cnn_layers:
                x = nn.Conv(
                    features=layers,
                    kernel_size=(3, 3),
                    dtype=self.dtype,
                    kernel_init=self.kernel_init,
                )
                x = nn.relu(x)
                x = self.batch_norm_cls(use_running_average=not train)(x)
        x = self.head(x)
        return x


def PredNet(
    cnn_layers,
    dense_layers,
    in_shape=(32, 32, 3),
    classes=10,
    transfer=True,
    base_model=None,
    lr=0.1,
    momentum=0.01,
):

    new_head = Head(dense_layers=dense_layers, num_classes=classes)

    return TransferModel(backbone=base_model, head=new_head, cnn_layers=cnn_layers)


def PredNet1(
    backbone_params=None,
    cnn_layers=2,
    dense_layers=1,
    in_shape=(32, 32, 3),
    classes=10,
    transfer=True,
    base_model=None,
    lr=0.1,
    momentum=0.01,
):

    new_head = Head(dense_layers=dense_layers, num_classes=classes)

    return TransferModel(
        backbone=base_model,
        backbone_params=backbone_params,
        head=new_head,
        cnn_layers=cnn_layers,
    )
