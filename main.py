import os
import argparse
import numpy as np
from tqdm import tqdm
from typing import Any

from dataloader import load_data
from RotNet import rotnet_constructor
from PredNet import prednet_constructor

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax import traverse_util
from flax.core.frozen_dict import freeze
from flax.training import train_state, checkpoints

import optax

# Define cifar10 image shape
CIFAR10_INPUT_SHAPE = (1, 32, 32, 3)

class TrainState(train_state.TrainState):
    batch_stats: Any

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rotnet_arch", type=str, default="rotnet3_feat3", help="RotNet architecture to use")
    parser.add_argument("--prednet_arch", type=str, default="prednet3", help="PredNet architecture to use")
    parser.add_argument("--rotnet_ckpt_dir", type=str, default="./ckpts/rotnet", help="directory to save RotNet checkpoints")
    parser.add_argument("--prednet_ckpt_dir", type=str, default="./ckpts/prednet", help="directory to save RotNet checkpoints")
    parser.add_argument("--transfer", action="store_true", default=False, help="load pretrained RotNet if set to True")
    parser.add_argument("--ckpt_epoch", default=0, type=int, metavar="N", help="epoch number to load RotNet checkpoint")
    parser.add_argument("--rotnet_epochs", default=10, type=int, metavar="N", help="number of RotNet epochs to run")
    parser.add_argument("--prednet_epochs", default=10, type=int, metavar="N", help="number of PredNet epochs to run")
    parser.add_argument("--batch_size", default=128, type=int, metavar="N", help="batch size per process")
    parser.add_argument("--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()
    return args

def cross_entropy_loss_(logits, labels, num_classes=10):
    """
    Define loss: https://flax.readthedocs.io/en/latest/getting_started.html#define-loss
    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=num_classes)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
cross_entropy_loss = jax.jit(cross_entropy_loss_, static_argnums=2)

def compute_metrics_(logits, labels, num_classes):
    """
    Metric computation: https://flax.readthedocs.io/en/latest/getting_started.html#metric-computation
    """
    loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {"loss": loss, "accuracy": accuracy}
    return metrics
compute_metrics = jax.jit(compute_metrics_, static_argnums=2)

def create_train_state(rng, model, learning_rate, momentum):
    """
    Create train state: https://flax.readthedocs.io/en/latest/getting_started.html#create-train-state
    """
    variables = model.init(rng, jnp.ones(CIFAR10_INPUT_SHAPE, dtype=model.dtype), train=False)
    params, batch_stats = variables["params"], variables["batch_stats"]
    tx = optax.sgd(learning_rate, momentum)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)
    return state, variables

def train_batch_(state, images, labels, num_classes=10):
    """
    Training step: https://flax.readthedocs.io/en/latest/getting_started.html#training-step
    """
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats}, images, mutable=["batch_stats"], train=True
        )
        loss = cross_entropy_loss(logits=logits, labels=labels, num_classes=num_classes)
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    metrics = compute_metrics(logits=logits, labels=labels, num_classes=num_classes)
    return state, metrics
train_batch = jax.jit(train_batch_, static_argnums=3)

def train_epoch(state, dataloader, num_classes=10):
    """
    Train function: https://flax.readthedocs.io/en/latest/getting_started.html#train-function
    """
    batch_metrics = []
    for images, labels in dataloader:
        state, metrics = train_batch(state, images, labels, num_classes=num_classes)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]
    }
    return state, epoch_metrics_np

def eval_batch_(state, images, labels, num_classes=10):
    """
    Evaluation step: https://flax.readthedocs.io/en/latest/getting_started.html#evaluation-step
    """
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats}, images, mutable=False, train=False
    )
    return compute_metrics(logits=logits, labels=labels, num_classes=num_classes)
eval_batch = jax.jit(eval_batch_, static_argnums=3)

def eval_model(state, dataloader, num_classes=10):
    """
    Eval function: https://flax.readthedocs.io/en/latest/getting_started.html#eval-function
    """
    batch_metrics = []
    for images, labels in dataloader:
        metrics = eval_batch(state, images, labels, num_classes=num_classes)
        batch_metrics.append(metrics)
    batch_metrics_np = jax.device_get(batch_metrics)
    validation_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]
    }
    return validation_metrics_np["loss"], validation_metrics_np["accuracy"]

def extract_submodule(model):
    feature_extractor = model.features.clone()
    variables = model.features.variables
    return feature_extractor, variables

def main():
    # ---------------------------- Parse the Arguments --------------------------- #
    args = parse()

    # ---------------------- Generate JAX Random Number Key ---------------------- #
    rng = jax.random.PRNGKey(0)
    print("Random Key Generated")

    # ------------------------------ Define Network ------------------------------ #
    # Define network: https://flax.readthedocs.io/en/latest/getting_started.html#define-network
    rotnet_model = rotnet_constructor(args.rotnet_arch)
    print("Network Defined")
    if args.verbose:
        print(nn.tabulate(rotnet_model, rng)(jnp.ones(CIFAR10_INPUT_SHAPE), False))

    # ------------------------- Load the CIFAR10 Dataset ------------------------- #
    # Loading data: https://flax.readthedocs.io/en/latest/getting_started.html#loading-data
    # NOTE: Choose batch_size and workers based on system specs.
    # NOTE: This dataloader requires pytorch to load the datset for convenience.
    loaders = load_data(batch_size=args.batch_size, workers=args.workers)
    train_loader, validation_loader, test_loader, rot_train_loader, rot_validation_loader, rot_test_loader = loaders
    print("Data Loaded")

    # --- Create the Train State Abstraction (see documentation in link below) --- #
    # Create train state: https://flax.readthedocs.io/en/latest/getting_started.html#create-train-state
    rotnet_state, rotnet_variables = create_train_state(rng, rotnet_model, args.lr, args.momentum)
    print("Train State Created")

    # ----------------- Specify the Directory to Save Checkpoints ---------------- #
    rotnet_ckpt_dir = args.rotnet_ckpt_dir
    if not os.path.exists(rotnet_ckpt_dir):
        os.makedirs(rotnet_ckpt_dir)
        print("RotNet Checkpoint Directory Created")
    else:
        print("RotNet Checkpoint Directory Found")

    # --------------- Load Existing Checkpoint of Pretrained RotNet -------------- #
    if args.transfer:
        rotnet_state = checkpoints.restore_checkpoint(
            ckpt_dir=rotnet_ckpt_dir, target=rotnet_state, step=args.ckpt_epoch
        )
        print("RotNet Checkpoint Loaded for Transfer Learning")

    # ------------------------ Otherwise Train the RotNet ------------------------ #
    else:
        print("Starting RotNet Training Loop")
        for epoch in tqdm(range(args.ckpt_epoch + 1, args.rotnet_epochs + 1)):
            # ------------------------------- Training Step ------------------------------ #
            # Training step: https://flax.readthedocs.io/en/latest/getting_started.html#training-step
            rotnet_state, train_epoch_metrics = train_epoch(
                rotnet_state, rot_train_loader, num_classes=4
            )

            # Print training metrics every epoch
            print(
                f"train epoch: {epoch}, \
                loss: {train_epoch_metrics['loss']:.4f}, \
                accuracy:{train_epoch_metrics['accuracy']*100:.2f}%"
            )

            # ------------------------------ Evaluation Step ----------------------------- #
            # Evaluation step: https://flax.readthedocs.io/en/latest/getting_started.html#evaluation-step
            validation_loss, _ = eval_model(rotnet_state, rot_validation_loader, num_classes=4)

            # Print validation metrics every epoch
            print(f"validation loss: {validation_loss:.4f}")

            # ---------------------------- Saving Checkpoints ---------------------------- #
            # ---- https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html --- #
            checkpoints.save_checkpoint(
                ckpt_dir=rotnet_ckpt_dir, target=rotnet_state, step=epoch, overwrite=True, keep=args.rotnet_epochs
            )

            # Print test metrics every nth epoch
            if epoch % 10 == 0:
                _, test_accuracy = eval_model(rotnet_state, rot_test_loader, num_classes=4)
                print(f"test_accuracy: {test_accuracy:.2f}")

    # ---- https://flax.readthedocs.io/en/latest/guides/transfer_learning.html --- #
    # ----------------------------- Extract Backbone ----------------------------- #
    backbone_model, backbone_model_variables = nn.apply(extract_submodule, rotnet_model)(rotnet_variables)
    
    # ------------------------- Create the Prednet Model ------------------------- #
    prednet_model = prednet_constructor(args.prednet_arch, backbone_model)
    
    # ----------------------- Extract Variables and Params ----------------------- #
    prednet_variables   = prednet_model.init(rng, jnp.ones(CIFAR10_INPUT_SHAPE), train=False)
    prednet_params      = prednet_variables['params']
    prednet_batch_stats = prednet_variables['batch_stats']
 
    # --------------------- Transfer the Backbone Parameters --------------------- #
    prednet_params              = prednet_params.unfreeze()
    prednet_params['backbone']  = backbone_model_variables['params']
    prednet_params              = freeze(prednet_params)
    
    prednet_batch_stats              = prednet_batch_stats.unfreeze()
    prednet_batch_stats['backbone']  = backbone_model_variables['batch_stats']
    prednet_batch_stats              = freeze(prednet_batch_stats)
    
    # -------------------------- Define How to Backprop -------------------------- #
    partition_optimizers = {'trainable': optax.sgd(args.lr, args.momentum), 'frozen': optax.set_to_zero()}
    prednet_param_partitions = freeze(traverse_util.path_aware_map(
        lambda path, _: 'frozen' if 'backbone' in path else 'trainable', prednet_params
    ))
    
    tx = optax.multi_transform(partition_optimizers, prednet_param_partitions)
    
    # ---------------- Visualize param_partitions to double check ---------------- #
    if args.verbose:
        flat = list(traverse_util.flatten_dict(prednet_param_partitions).items())
        freeze(traverse_util.unflatten_dict(dict(flat[:2] + flat[-2:])))
    
    # ---------------------- Create Train State for PredNet ---------------------- #
    prednet_state = TrainState.create(
        apply_fn=prednet_model.apply, params=prednet_params, tx=tx, batch_stats=prednet_batch_stats
    )
    
    # ----------------- Specify the Directory to Save Checkpoints ---------------- #
    prednet_ckpt_dir = args.prednet_ckpt_dir
    if not os.path.exists(prednet_ckpt_dir):
        os.makedirs(prednet_ckpt_dir)
        print("PredNet Checkpoint Directory Created")
    else:
        print("PredNet Checkpoint Directory Found")
    
    # ----------------------------- Train the PredNet ---------------------------- #
    print("Starting PredNet Training Loop")
    for epoch in tqdm(range(1, args.prednet_epochs + 1)):
        # ------------------------------- Training Step ------------------------------ #
        # Training step: https://flax.readthedocs.io/en/latest/getting_started.html#training-step
        prednet_state, train_epoch_metrics = train_epoch(
            prednet_state, train_loader, num_classes=10
        )

        # Print training metrics every epoch
        print(
            f"train epoch: {epoch}, \
            loss: {train_epoch_metrics['loss']:.4f}, \
            accuracy:{train_epoch_metrics['accuracy']*100:.2f}%"
        )

        # ------------------------------ Evaluation Step ----------------------------- #
        # Evaluation step: https://flax.readthedocs.io/en/latest/getting_started.html#evaluation-step
        validation_loss, _ = eval_model(prednet_state, validation_loader, num_classes=10)

        # Print validation metrics every epoch
        print(f"validation loss: {validation_loss:.4f}")

        # ---------------------------- Saving Checkpoints ---------------------------- #
        # ---- https://flax.readthedocs.io/en/latest/guides/use_checkpointing.html --- #
        checkpoints.save_checkpoint(
            ckpt_dir=prednet_ckpt_dir, target=prednet_state, step=epoch, overwrite=True, keep=args.prednet_epochs
        )

        # Print test metrics every nth epoch
        if epoch % 10 == 0:
            _, test_accuracy = eval_model(prednet_state, test_loader, num_classes=10)
            print(f"test_accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    main()
