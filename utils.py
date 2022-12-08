"""
This file is almost entirely based on the example here: 
https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

"""
import numpy as np
import jax.numpy as jnp
from torch.utils import data

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.array(pic.permute(1, 2, 0),dtype=jnp.float32)

def rotate_image(images):
    """
        This function takes in a numpy array of shape (batch_size, img_l, img_w, num_channels)
        and returns a numpy array of shape (4*batch_size, img_l, img_w, num_channels) with the
        rotated copies of each image and rotation labels of length (4*batch_size).
    """
    batch_size, _, _, _ = images.shape
    images_90  = jnp.rot90(images, 1, (-3,-2))
    images_180 = jnp.rot90(images, 2, (-3,-2))
    images_270 = jnp.rot90(images, 3, (-3,-2))

    # ------------------------- Stack the rotated images ------------------------- #
    rotated_image_set = jnp.vstack((images, images_90.copy(), images_180.copy(), images_270.copy()))

    # ------------------------ Create the rotation labels ------------------------ #
    rotation_labels = jnp.hstack((jnp.zeros(batch_size), jnp.ones(batch_size), 2*jnp.ones(batch_size), 3*jnp.ones(batch_size)))

    return np.array(rotated_image_set), np.array(rotation_labels)
  