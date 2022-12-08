import copy
import torch.utils
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from utils import NumpyLoader, FlattenAndCast, rotate_image

# Define directory to load data from
CIFAR_DATA_DIR = "./datasets/CIFAR"

"""
Many sections inspired by code here: https://github.com/fattorib/Flax-ResNets
"""
def load_data(batch_size=128, workers=4):
    
    # ---------------- Define the Transforms For Loading the Data ---------------- #
    # NOTE: You can add options here for more training time transforms.
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
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

    rot_cls = ['0', '90', '180', '270']
    rot_cls_to_idx = {
        "0": 0,
        "90": 1,
        "180": 2,
        "270": 3,
    }

    # -------------------- Load the Datasets in Pytorch Format ------------------- #
    train_dataset       = CIFAR10(root=CIFAR_DATA_DIR, train=True, download=True, transform=transform_train)
    
    # ---------------- Create the Self-Supervised Rotation Dataset --------------- #
    rot_train_dataset   = copy.deepcopy(train_dataset)

    rot_train_dataset.data, targets     = rotate_image(rot_train_dataset.data)
    rot_train_dataset.targets           = targets.tolist()
    rot_train_dataset.class_to_idx      = copy.deepcopy(rot_cls_to_idx)
    rot_train_dataset.classes           = copy.deepcopy(rot_cls)

    # ----------- Now Split the Loaded Dataset into Train and Validation ---------- #
    # NOTE: Here we use a 90/10 split. You can change the split below as per requirement.
    train_dataset, validation_dataset           = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    rot_train_dataset, rot_validation_dataset   = torch.utils.data.random_split(rot_train_dataset, [0.9, 0.1])

    # ----------------------------- Load the Test Set ---------------------------- #
    test_dataset        = CIFAR10(root=CIFAR_DATA_DIR, train=False, download=True, transform=transform_test)
    
    # ---------------- Create the Self-Supervised Rotation Dataset --------------- #
    rot_test_dataset    = copy.deepcopy(test_dataset)

    rot_test_dataset.data, targets     = rotate_image(rot_test_dataset.data)
    rot_test_dataset.targets           = targets.tolist()
    rot_test_dataset.class_to_idx      = copy.deepcopy(rot_cls_to_idx)
    rot_test_dataset.classes           = copy.deepcopy(rot_cls)

    # --------- Now Convert the Loaded Pytorch Dataset to Jax/Flax Format -------- #
    train_loader = NumpyLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )

    rot_train_loader = NumpyLoader(
        rot_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = NumpyLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False,
    )

    rot_validation_loader = NumpyLoader(
        rot_validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = NumpyLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False,
    )

    rot_test_loader = NumpyLoader(
        rot_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------- Return the Image Loaders ------------------------- #
    return train_loader, validation_loader, test_loader, rot_train_loader, rot_validation_loader, rot_test_loader
