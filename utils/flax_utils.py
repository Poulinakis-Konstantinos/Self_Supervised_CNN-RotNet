"""
This file is almost entirely based on the example here: 
https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

Last Accessed: 5th December 2022
"""
import numpy as np
from torch.utils import data
import jax.numpy as jnp
import jax
import flax

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