import argparse
import functools
from typing import Any
from jax._src.dtypes import dtype
import jax.numpy as jnp
import jax
import torchvision.transforms as transforms
from RotNet import RotNet3, RotNet4, RotNet5
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
import torch.utils
import wandb
import flax
from utils.dataloader import load_data

class TrainState(train_state.TrainState):
    batch_stats: Any = None
    weight_decay: Any = None
    dynamic_scale: flax.optim.DynamicScale = None


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data", "--data", default="ML/", type=str, metavar="DIR", help="path to dataset")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=180, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size per process (default: 128)")
    parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)")
    parser.add_argument("--model", type=str, default="ResNet20")
    parser.add_argument("--CIFAR10", type=bool, default=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="fp32")
    args = parser.parse_args()
    return args

def create_train_state(rng, model, learning_rate, momentum):
    variables = model.init(rng, jnp.ones((1, 32, 32, 3), dtype=model.dtype), train=True)
    params, batch_stats = variables['params'], variables['batch_stats']
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

def main():
    args = parse()

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    model = RotNet3()
    train_loader, validation_loader, test_loader = load_data()

    state = create_train_state(rng, model, args.lr, args.momentum)
