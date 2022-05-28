import tensorflow as tf
import keras
from keras import layers, activations, optimizers, Sequential
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




# def RotNet(classes=4) :
#     ''' Define the RotNet model architecture. '''
#     model = Sequential()
#
#     model.add(layers.Input(shape=(32, 32, 3)))
#     model.add(layers.Conv2D(96, (12,12), activation='relu'))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Conv2D(192, (6,6), activation='relu'))
#     model.add(layers.BatchNormalization() )
#     model.add(layers.Conv2D(192, (6,6), activation='relu'))
#     model.add(layers.BatchNormalization() )
#     model.add(layers.Conv2D(192, (3,3), activation='relu'))
#     model.add(layers.BatchNormalization() )
#
#     model.add(layers.Flatten() )
#     model.add(layers.Dense(200, activation='relu'))
#     model.add(layers.BatchNormalization() )
#     model.add(layers.Dense(200, activation='relu'))
#     model.add(layers.BatchNormalization() )
#
#     model.add(layers.Dense(classes, activation='softmax'))
#
#     model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
#     return model


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)
    EPOCHS = 10  # default value
    #epochs = int(sys.argv[1])

    # create the model object
    # cnn and dense options
    cnn_layers = [(96, 12), (192, 6), (192, 6), (192, 3)]
    dense_layers = [200, 200]

    rotnet = RotNet(cnn_layers, dense_layers,
                    num_classes=4, input_shape=(32, 32, 3))
    rotnet.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    # print("Model Archtecture :")
    # print(rotnet.summary())
    # Load data
    (X, y_train), _ = loaders.load_data()
    # np.array to tensor
    x_train = tf.convert_to_tensor(X, dtype=tf.int32)
    x_train = tf.reshape(x_train, (-1, 32, 32, 3))
    print("Input shape : ", x_train.shape)

    # Train the model via self-supervision
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
    history = self_supervised_trainer(rotnet, x_train[0:1000], EPOCHS,
                                    optimizer, batch_size=32, val_split=0.1, shuffle=True)
    plot_training_curves(history)
    # rotnet.save_weights('self_supervised.h5')
    # rotnet.load_weights('self_supervised.h5')

    # Save the entire model
    rotnet.save('saved_model/rotnet_model')

    # Evaluate on a test set
    test_set = x_train[2000:2200]
    aug_test, y_test = rotate_image(test_set)
    y_preds = rotnet(aug_test)
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_accuracy.update_state(y_test, y_preds)
    print(f"Test set accuracy is {epoch_accuracy.result()}")
    # Test set loss
    test_loss = get_loss(rotnet, aug_test, y_test)
    print(f"Test set loss : {test_loss}")

    print("Labels test set sample :", np.argmax(y_preds[0:24], axis=1))
    plot_sample(aug_test[0: 24], 6, 4)
