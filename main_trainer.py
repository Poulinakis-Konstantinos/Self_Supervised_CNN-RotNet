# TODO: Clean this up!
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
from utils.flax_utils import rotate_image
from tqdm import tqdm
from flax.training import train_state, checkpoints
import os

class TrainState(train_state.TrainState):
    batch_stats: Any

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
    parser.add_argument("-b", "--batch_size", default=128, type=int, metavar="N", help="mini-batch size per process (default: 128)")
    parser.add_argument("--model_rotnet", type=str, default="RotNet3")
    parser.add_argument("--model_prednet", type=str, default="PredNet3")
    parser.add_argument("--CIFAR10", type=bool, default=True)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="fp32")
    args = parser.parse_args()
    return args

def cross_entropy_loss_(logits, labels, num_classes=10):
    """
        Step 3: https://flax.readthedocs.io/en/latest/getting_started.html#define-loss
    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
cross_entropy_loss = jax.jit(cross_entropy_loss_, static_argnums=2)

def compute_metrics_(logits, labels, num_classes):
    """
        Step 4: https://flax.readthedocs.io/en/latest/getting_started.html#metric-computation
    """
    loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics
compute_metrics = jax.jit(compute_metrics_, static_argnums=2)

def create_train_state(rng, model, learning_rate, momentum):
    """
        Step 6: https://flax.readthedocs.io/en/latest/getting_started.html#create-train-state
    """
    variables = model.init(rng, jnp.ones((1, 32, 32, 3), dtype=model.dtype), train=False)
    params, batch_stats = variables['params'], variables['batch_stats']
    tx = optax.sgd(learning_rate, momentum)
    # TODO: Check if this is correct for BatchNorm
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

def train_batch_(state, images, labels, num_classes=10):
    """
        Step 7: https://flax.readthedocs.io/en/latest/getting_started.html#training-step
    """
    def loss_fn(params):
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats}, images, mutable=["batch_stats"], train=True
        )

        loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)
        
        return loss, (logits, new_state)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (logits, new_state)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_state['batch_stats'])
    metrics = compute_metrics(logits=logits, labels=labels, num_classes=num_classes)
    return state, metrics
train_batch = jax.jit(train_batch_, static_argnums=3)

def train_epoch(state, dataloader, num_classes=10):
    """
        Step 9: https://flax.readthedocs.io/en/latest/getting_started.html#train-function
    """
    # TODO: Shuffle Please!!
    batch_metrics = []
    for images, labels in dataloader:
        # --------- Change the labels and modify batch for backbone training --------- #
        state, metrics = train_batch(state, images, labels, num_classes=num_classes)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return state, epoch_metrics_np

def eval_batch_(state, images, labels, num_classes=10):
    """
        Step 8: https://flax.readthedocs.io/en/latest/getting_started.html#evaluation-step
    """
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, images, mutable=False, train=False
    )
    return compute_metrics(logits=logits, labels=labels, num_classes=num_classes)
eval_batch = jax.jit(eval_batch_, static_argnums=3)

def eval_model(state, dataloader, num_classes=10):
    """
        Step 10: https://flax.readthedocs.io/en/latest/getting_started.html#eval-function
    """
    batch_metrics = []
    for images, labels in dataloader:
        metrics = eval_batch(state, images, labels, num_classes=num_classes)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    validation_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return validation_metrics_np["loss"], validation_metrics_np["accuracy"]

def main():
    args = parse()

    # ---------------------- Generate JAX Random Number Key ---------------------- #
    rng = jax.random.PRNGKey(0)
    rng, _ = jax.random.split(rng)
    print("Random Gen Complete")

    # ------------------------------ Define network ------------------------------ #
    # Step 2: https://flax.readthedocs.io/en/latest/getting_started.html#define-network
    # TODO: Fix This!
    model = RotNet3()

    print("Network Defined")

    # ------------------------- Load the CIFAR10 dataset ------------------------- #
    # Step 5: https://flax.readthedocs.io/en/latest/getting_started.html#loading-data
    # NOTE: Choose batch_size and workers based on system specs
    # NOTE: This dataloader requires pytorch to load the datset for convenience.
    train_loader, validation_loader, test_loader, rot_train_loader, rot_validation_loader, rot_test_loader = load_data(batch_size=args.batch_size, workers=args.workers)
    print("Data Loaded!")
    # --- Create the Train State Abstraction (see documentation in link below) --- #
    # Step 6: https://flax.readthedocs.io/en/latest/getting_started.html#create-train-state
    state = create_train_state(rng, model, args.lr, args.momentum)
    print("Train State Created")
    
    # createing state directory
    path = "./state_root"
    if not os.path.exists(path):
        os.makedirs(path)
        print("creating root directory for state")
    else:
        print("find existing root directory for state")
    
    print("Starting Training Loop!")
    for epoch in tqdm(range(args.epochs)):

        # ------------------------------- Training Step ------------------------------ #
        # Step 7: https://flax.readthedocs.io/en/latest/getting_started.html#training-step
        state, train_epoch_metrics_np = train_epoch(state, rot_train_loader, num_classes=4)
        # ---------------------------- Saving Checkpoints ---------------------------- #
        # ---- https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html --- #
        checkpoints.save_checkpoint(ckpt_dir="./ckpts", target=state, step=epoch)

        # Print train metrics every epoch
        print(
            f"train epoch: {epoch}, \
            loss: {train_epoch_metrics_np['loss']:.4f}, \
            accuracy:{train_epoch_metrics_np['accuracy']*100:.2f}%"
        )

        # ------------------------------ Evaluation Step ----------------------------- #
        #  Step 8: https://flax.readthedocs.io/en/latest/getting_started.html#evaluation-step
        validation_loss, _ = eval_model(state, rot_validation_loader, num_classes=4)

        # Print validation metrics every epoch
        print(f"validation loss: {validation_loss:.4f}")
        
        if epoch % 10 == 0:
            # Print test metrics every nth epoch
            _, test_accuracy = eval_model(state, rot_test_loader, num_classes=4)
            print(f"test_accuracy: {test_accuracy:.2f}")


if __name__ == '__main__':
    main()
