from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, activations, optimizers, Sequential, regularizers
import numpy as np
from utils import plot_training_curves, plot_sample
from trainers import rotate_image


class CnnBlock(layers.Layer):
    '''
    Basic CnnBlock v.1.0
    '''

    def __init__(self, out_channels, kernel):
        super(CnnBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = activations.relu(x)
        return x


class RotNet(keras.Model):
    def __init__(self, cnn_layers, dense_layers, num_classes=4, input_size=(32, 32, 3)):
        super(RotNet, self).__init__()
        self.num_classes = num_classes
        self.cnns = []
        self.denses = []
        self.input_size = input_size
        # cnn layers
        for layer in cnn_layers:
            self.cnns.append(
                CnnBlock(layer[0], layer[1])
            )
        self.flatten = layers.Flatten()
        for layer in dense_layers:
            self.denses.append(
                layers.Dense(layer, activation='relu')
            )
        # layer for rotation classification
        self.output_layer = layers.Dense(num_classes)

    def call(self, input_tensor, training=False):
        x = tf.identity(input_tensor)
        # cnn layers
        for cnn in self.cnns:
            x = cnn(x, training=training)
        # flatten
        x = layers.Flatten()(x)
        # dense layers
        for dense in self.denses:
            x = dense(x)
        # output_layer
        x = self.output_layer(x)
        return x

    def eval_on_rotations(self, x_test, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)) :
        '''Evaluates on a test set of images by rotating them and then feeding them
        Parameters :
            x_test (tensors) : The test images shape (-1, im_size, im_size, channels)
            loss  : The loss function (default SparseCategoricalCrossEntropy)
        Returns :
            acc (float32) : accuracy on test set
            loss (float32) : loss on test set'''

        aug_test, y_test = rotate_image(x_test)
        y_preds = self.call(aug_test, training=False)
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        epoch_accuracy.update_state(y_test, y_preds)

        # Test set loss
        preds = self.call(aug_test)
        test_loss = loss(y_true=y_test, y_pred=preds)
        print(f"Test set accuracy is {epoch_accuracy.result()}")
        print(f"Test set loss : {test_loss}")

        print("Labels test set sample :", np.argmax(y_preds[0:4], axis=1))


def RotNet_constructor(build_instructions: dict):
    '''
    This is a constructor of a specific rotnet model.
    Given a set of appropriate build_instructions in dictionary form,
    it produces a specific keras.Model that has the intended architecture...
    '''

    inputs = keras.Input(shape=tuple(build_instructions['input_shape']))
    x = tf.identity(inputs)
    #cnn layers...
    for layer in build_instructions['cnn_layers']:
        x = layers.Conv2D(layer[0], layer[1])(x)
        x = layers.BatchNormalization()(x)
        x = keras.activations.relu(x)
        if build_instructions['include_maxpool']:
            x = layers.MaxPooling2D()(x)
    #flatten...
    x = layers.Flatten()(x)
    #dense layers...
    for layer in build_instructions['dense_layers']:
        x = layers.Dense(layer, activation = 'relu')(x)
    #output layer...
    x = layers.Dense(build_instructions['num_classes'])(x)

    return keras.Model(inputs = inputs, outputs = x, name=build_instructions['name'])
