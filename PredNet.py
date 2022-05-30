import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, optimizers, Sequential, regularizers
from loaders import  load_dataset
from utils import plot_training_curves,plot_sample
from RotNet import train_loop,get_loss
import numpy as np
from os import path 


SAVE_PATH = 'Saved_models'

def PredNet(cnn_layers, dense_layers, in_shape=(32,32,3), classes=10, transfer=True, base_model_name='rotnet_v1'):
    ''' Define the PredNet model architecture. '''

    model = Sequential()

    if (transfer):
        backbone = tf.keras.models.load_model(path.join(SAVE_PATH, base_model_name))
        backbone_model = Sequential(backbone.layers[0:2*len(cnn_layers)])
        model.add(backbone_model)
    else:
        # ============ Backbone ================
        model.add(keras.Input(shape=in_shape))

        # Convolution - Batch Norm layers
        for layer in cnn_layers:
            out_channels, kernel = layer
            model.add(layers.Conv2D(out_channels, kernel, activation='relu'))
            model.add(layers.BatchNormalization())

       # ========================================

    model.add(layers.Flatten())

    for layer in dense_layers:
        model.add(layers.Dense(layer, activation='relu'))
        model.add(layers.BatchNormalization())

    model.add(layers.Dense(classes, activation='softmax'))

    return model

def PredNet_constructor(build_instructions: dict):
    '''
    This is a constructor of a specific prednet model.
    Given a set of appropriate build_instructions in dictionary form,
    it produces a specific keras.Model that has the intended architecture...
    '''

    #loads a specific rotnet model...
    rotnet = keras.models.load_model(build_instructions['load_from'])
    #keeps until given layer. For example -2 for keeping up to the -2 layer.
    #attention on what layers to load is demanded!!!
    x = rotnet.layers[build_instructions['keep_until']].output
    #typical construction...
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

    return keras.Model(inputs = rotnet.input, outputs = x, name=build_instructions['name'])

