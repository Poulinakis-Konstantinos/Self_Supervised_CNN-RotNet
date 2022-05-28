from loaders import Img_dloader, load_data
from RotNet import RotNet
from trainers import self_supervised_trainer, supervised_trainer
from utils import plot_sample, plot_training_curves

from os import path
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adadelta
import matplotlib.pyplot as plt

SAVE_PATH = 'Saved_models'

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()
    #samples = x_train[0:4]
    #plot_sample(samples, 2, 2)

    # Check execution device
    print(tf.config.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)

    # Model Hyperparameters
    cnn_layers = [(96, 12), (192, 6), (192, 6), (192, 3)]
    dense_layers = [200, 200]
    num_classes = 4
    IMG_SHAPE = [32, 32, 3]
    # Training Hyperparameters
    EPOCHS = 1  # default value
    LR = 0.1
    BATCH_SIZE = 32
    VAL_SPLIT = 0.1
    optimizer = Adadelta(learning_rate=LR)
    loss_function = SparseCategoricalCrossentropy(from_logits=True)

    # Convert data to tensors of float32 type
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    # create RotNet Model (self-supervised)
    rotnet = RotNet(cnn_layers, dense_layers, num_classes=num_classes, input_size=IMG_SHAPE)
    history = self_supervised_trainer(rotnet, x_train[0:10], EPOCHS,
                                      optimizer, batch_size=BATCH_SIZE,
                                      val_split=VAL_SPLIT, shuffle=True)
    print(rotnet.summary())
    plot_training_curves(history)
    # Evaluate the RotNet based on rotation prediction
    rotnet.eval_on_rotations(x_test[0:10])

    # Save the entire RotNet model
    MODEL_NAME = 'rotnet_v1'
    rotnet.save(path.join(SAVE_PATH, MODEL_NAME))