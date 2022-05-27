import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Sequential
from time import time
from  tqdm import tqdm
import sys


def backbone(model_path) :
    rotnet_model = keras.models.load_model(model_path)
    # Cut the linear layers from rotnet and extract the backbone model
    backbone= model.layers[-6].output

    return backbone


BACKBONE = 'saved_model/rotnet_model'

back_model
