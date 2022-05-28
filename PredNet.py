import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import load_dataset
from utils import plot_training_curves,plot_sample
from RotNet import train_loop,get_loss
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

if __name__=='__main__' :

    # Check execution device
    print(tf.config.list_physical_devices('GPU'))
    #tf.debugging.set_log_device_placement(True)

    # Hyperparameters
    epochs = 100 # default value
    cnn_layers = [(96, 12),(192, 6),(192, 6),(192, 3)]
    dense_layers = [200, 200]
    num_classes = 10
    in_shape = [32, 32, 3]
    lr = 0.1
    batch_size = 32
    val_split = 0.1
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=lr)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    transfer = True

    # Initialize model
    prednet = PredNet(cnn_layers,dense_layers,in_shape,classes=num_classes,transfer = transfer)
    prednet.compile(optimizer=optimizer,loss=loss_fn, metrics=['accuracy'])
    prednet.build(input_shape=[None] + in_shape)

    # Print Model summary
    if (transfer):
        backbone = prednet.layers[0]
        print(backbone.summary())
    print(prednet.summary())

    # Load and convert data
    (X, y), _ = load_dataset.load_data()
    x = tf.convert_to_tensor(X, dtype=tf.int32)  # np.array to tensor
    x = tf.reshape(x, (-1, 32, 32, 3))

    # Train model
    history = train_loop(model = prednet, x = x, y = y, epochs= epochs, optimizer = optimizer, batch_size=batch_size, val_split=val_split, shuffle=True)
    plot_training_curves(history)

    # Evaluate on a test set
    x_test = x[2000:2200]
    y_test = y[2000:2200]
    y_preds = prednet(x_test)
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy.update_state(y_test, y_preds)
    print(f"Test set accuracy is {epoch_accuracy.result()}")
    # Test set loss
    test_loss = get_loss(prednet, x_test, y_test)
    print(f"Test set loss : {test_loss}")

    print("Labels test set sample :", np.argmax(y_preds[0:24], axis=1))
    plot_sample(x_test[0: 24], 6, 4)