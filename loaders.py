import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    return (x_train, y_train), (x_test, y_test)


# class Img_dloader(keras.utils.Sequence):
#     """Helper to iterate over the data (as Numpy arrays)."""

#     def __init__(self, batch_size, input_img, img_size):
#         self.batch_size = batch_size
#         self.img_size = img_size
#         self.input_imgs = input_img

#     def __len__(self):
#         return len(self.img_size) // self.batch_size

#     def __getitem__(self, idx):
#         """Returns tuple (input, target) correspond to batch #idx."""
#         i = idx * self.batch_size
#         batch_input_img = self.input_imgs[i: i + self.batch_size]

#         x = np.zeros((self.batch_size,) + (self.img_size,self.img_size) + (3,), dtype="float32")

#         for j, img in enumerate(batch_input_img):
#             x[j] = img
#         print("DL ", idx, img.shape)

#         return x
