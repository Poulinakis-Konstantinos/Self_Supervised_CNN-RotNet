import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.utils

from utils.flax_utils import NumpyLoader, FlattenAndCast

CIFAR_DATA_DIR = "./datasets/CIFAR"

"""
Many sections inspired by code here: https://github.com/fattorib/Flax-ResNets
"""
def load_data(batch_size=128, workers=4):
    
    # ---------------- Define the Transforms For Loading the Data ---------------- #
    # TODO: Add option here for more train time transforms.
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.RandomCrop(
            #     (32, 32),
            #     padding=4,
            #     fill=0,
            #     padding_mode="constant",
            # ),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
            FlattenAndCast(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            ),
            FlattenAndCast(),
        ]
    )

    # -------------------- Load the Datasets in Pytorch Format ------------------- #
    train_dataset = CIFAR10(root=CIFAR_DATA_DIR, train=True, download=True, transform=transform_train)

    # ----------- Now split the loaded dataset into tran and validation ---------- #
    # NOTE: Here we use a 90/10 split. YOu can change the split below as per requirement.
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

    # ----------------------------- Load the Test Set ---------------------------- #
    test_dataset = CIFAR10(root=CIFAR_DATA_DIR, train=False, download=True, transform=transform_test)
    
    # --------- Now convert the loaded pytorch dataset to Jax/Flax format -------- #
    train_loader = NumpyLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )

    validation_loader = NumpyLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )

    test_loader = NumpyLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )

    # ------------------------- Return the Image Loaders ------------------------- #
    return train_loader, validation_loader, test_loader
