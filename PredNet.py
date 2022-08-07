import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, optimizers, Sequential, regularizers
from loaders import load_data
from utils import plot_training_curves,plot_sample
import numpy as np

def PredNet(cnn_layers,dense_layers,in_shape,classes=10,transfer = True) :
    ''' Define the PredNet model architecture. '''

    model = Sequential()

    if (transfer):
        backbone = tf.keras.models.load_model("saved_model/RotNet")
        backbone_model = Sequential(backbone.layers[0:2*len(cnn_layers)])
        model.add(backbone_model)
    else:
        #============ Backbone ================
        model.add(keras.Input(shape=in_shape))

        # Convolution - Batch Norm layers
        for layer in cnn_layers:
            out_channels, kernel = layer
            model.add(layers.Conv2D(out_channels, kernel, activation='relu'))
            model.add(layers.BatchNormalization())

       #========================================

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
    if(build_instructions['transfer']):
        x = rotnet.layers[build_instructions['keep_until']].output
    else:
        inputs = keras.Input(shape=tuple(build_instructions['input_shape']))
        x = tf.identity(inputs)
        # cnn layers...
        for layer in build_instructions['backbone_cnn_layers']:
            x = layers.Conv2D(layer[0], layer[1])(x)
            x = layers.BatchNormalization()(x)
            x = keras.activations.relu(x)
            if build_instructions['include_maxpool']:
                x = layers.MaxPooling2D()(x)

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

    if (build_instructions['transfer']):
        return keras.Model(inputs = rotnet.input, outputs = x, name=build_instructions['name'])
    else:
        return keras.Model(inputs=inputs, outputs=x, name=build_instructions['name'])