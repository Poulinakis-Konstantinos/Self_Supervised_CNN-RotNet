from load_dataset import load_data
from RotNet import rotate_image
import matplotlib.pyplot as plt
from utils import plot_sample

if __name__=='__main__' :
    (x_train, y_train), (x_test, y_test) = load_data()
    samples = x_train[0:4]
    plot_sample(samples, 2, 2)

    ims, ims90, ims180, ims270  = rotate_image(samples)
    plot_sample(ims90, 2, 2)
    plot_sample(ims180, 2, 2)
    plot_sample(ims270, 2, 2)

