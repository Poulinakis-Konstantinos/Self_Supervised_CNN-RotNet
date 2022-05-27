from sklearn.model_selection import learning_curve
import tensorflow as tf
from keras import layers, optimizers, Sequential
import numpy as np
from scipy.ndimage import rotate
import load_dataset
from time import time
from  tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import plot_training_curves, plot_sample


def rotate_image(images) :
    ''' Rotate the given image by 0, 90, 180, 270 degrees
    images(tensor) : A tensor of shape (num_imgs, num_rows, num_columns, num_channels) '''

    rotated_imgs = []
    for image in images :
        rotated_imgs.append(image)
        rotated_imgs.append(rotate( image, 90) )
        rotated_imgs.append(rotate( image, 180) )
        rotated_imgs.append(rotate( image, 270) )
    # the labels of rotations
    rotations = [0, 1, 2, 3] * images.shape[0]
    return np.array(rotated_imgs), np.array(rotations)

def train_test_split_tensors(X, y, **options):
    """
    encapsulation for the sklearn.model_selection.train_test_split function
    in order to split tensors objects and return tensors as output

    :param X: tensorflow.Tensor object
    :param y: tensorflow.Tensor object
    :dict **options: typical sklearn options are available, such as test_size and train_size
    """

    if y==None :
        X_train, X_test = train_test_split(X.numpy(),  **options)
        X_train, X_test = tf.constant(X_train), tf.constant(X_test)
        return X_train, X_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(),  **options)
        X_train, X_test = tf.constant(X_train), tf.constant(X_test)
        y_train, y_test = tf.constant(y_train), tf.constant(y_test)
        return X_train, X_test, y_train, y_test
    return print("Invalid y value. Should be None or list or np.array")

class CnnBlock(layers.Layer):
    '''
    Basic CnnBlock v.1.0
    '''
    def __init__(self, out_channels, kernel):
        super(CnnBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel)
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training = False):
        x = self.conv(input_tensor)
        x = self.bn(x, training = training)
        x = keras.activations.relu(x)
        return x

class RotNet(keras.Model):
    def __init__(self, cnn_layers, dense_layers, num_classes = 4, input_shape = (32,32,3)):
        super(prednet, self).__init__()
        self.num_classes = num_classes
        self.cnns = []
        self.denses = []
        self.input_shape = input_shape
        #cnn layers
        for layer in cnn_layers:
            self.cnns.append(
                CnnBlock(layer[0], layer[1])
            )
        self.flatten = layers.Flatten()
        for layer in dense_layers:
            self.denses.append(
                layers.Dense(layer,activation = 'relu')
            )
        self.output_layer = layers.Dense(num_classes)

    def call(self, input_tensor, training = False):
        x = tf.identity(input_tensor)
        #cnn layers
        for cnn in self.cnns:
            x = cnn(x, training = training)
        #flatten
        x = layers.Flatten()(x)
        #dense layers
        for dense in self.denses:
            x = dense(x)
        #output_layer
        x = self.output_layer(x)
        return x

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

def get_loss(model, x, y, training=False,
             f_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)) :
    ''' Calculate loss value between ground truth y and model's output, on input x . '''
    y_ = model(x, training=training)
    return f_loss(y_true=y, y_pred=y_)

def grad(model, inputs, targets,f_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)) :

    with tf.GradientTape() as tape :
        l_value = get_loss(model, inputs, targets, training=True, f_loss=f_loss)
    # loss's gradients over all model parameters.
    grads = tape.gradient(l_value, model.trainable_variables)
    return l_value, grads

def train_loop(model, x, y, epochs, optimizer, batch_size=32, val_x=None, val_y=None, val_split=0.2, shuffle=True) :
    ''' Custom training loop '''
    # lists to store values for visualization
    tr_loss = []
    val_loss =[]
    tr_acc = []
    val_acc = []

    if val_x==None and val_y==None :
        # create validation split
        x_train, x_val, y_train, y_val = train_test_split_tensors(x, tf.convert_to_tensor(y, tf.int32),
                                                                    test_size=val_split, shuffle=shuffle)
    else :
        x_train=x ; y_train=y ; x_val=val_x ; y_val=val_y

    print(f"Initializing Training for {epochs} epochs")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in tqdm(range(epochs)) :
        # Training loop - using batches of batch_size
        for index, offset in enumerate(range(0, x_train.shape[0], batch_size)) :
            if (offset + batch_size < x_train.shape[0]) : upper = offset + batch_size   # avoid out of bounds error
            else : upper = x_train.shape[0]
            x_batch = x_train[offset : upper] #  creating batches of size batch_size
            y_batch = y_train[offset : upper]
            # calculate loss (forward pass) and gradients (backward pass)
            loss_value, grads = grad(model, x_batch, y_batch)
            # apply weight updates using the optimizer's algorithm
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Update progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_accuracy.update_state(y_batch, model(x_batch, training=True))
        # epoch's end
        tr_loss.append(epoch_loss_avg.result().numpy())
        tr_acc.append(epoch_accuracy.result().numpy())

        # Compute validation loss and accuracy
        val_loss.append(get_loss(model, x_val, y_val, training=False).numpy())
        epoch_accuracy.reset_state()       # clean training accuracy data to compute val_accuracy
        epoch_accuracy.update_state(y_val, model(x_val, training=False))
        val_acc.append(epoch_accuracy.result().numpy())
        # reset states
        epoch_accuracy.reset_state()
        epoch_loss_avg.reset_state()

        # print loss
        print("\n")
        print("Epoch: {}/{},  Train_Loss: {:9.4f}   Train_acc: {:9.4f} ,   Val_Loss: {:9.4f}   Val_acc: {:9.4f} ".format(epoch,
              epochs, float(tr_loss[epoch]), float(tr_acc[epoch]), float(val_loss[epoch]), float(val_acc[epoch]) ))

    return ((tr_loss, val_loss), (tr_acc, val_acc))

def self_supervised_train(model, x, epochs, optimizer, batch_size=32, val_x=None, val_y=None, val_split=0.2, shuffle=True) :
    ''' Custom training loop '''
    # lists to store values for visualization
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []

    if val_x==None and val_y==None :
        # create validation split
        x_train, x_val = train_test_split_tensors(x, y=None, test_size=val_split, shuffle=shuffle)
    else :
        x_train=x ; x_val=val_x

   # plot_sample(x_train[0:16], 4, 4)
    # create the augmented validation data and the val rotation labels
    x_val, y_val = rotate_image(x_val)

    print(f"Initializing Self-Supervised Training for {epochs} epochs")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in tqdm(range(epochs)):
        # Training loop - using batches of batch_size
        for index, offset in enumerate(range(0, x_train.shape[0], batch_size)) :
            if (offset + batch_size < x_train.shape[0]) : upper = offset + batch_size   # avoid out of bounds error
            else : upper = x_train.shape[0]
            x_batch = x_train[offset : upper] #  creating batches of size batch_size
           # plot_sample(x_batch[0:16], 4, 4)
            # Create rotated images and rotation labels
            augmented_x, rot_label = rotate_image(x_batch)
          #  plot_sample(augmented_x[0:16], 4, 4)
            # Shuffle the batch before feeding
            # indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
            # shuffled_indices = tf.random.shuffle(indices)
            # shuffled_x = tf.gather(x, shuffled_indices)
            # shuffled_y = tf.gather(y, shuffled_indices)

            # calculate loss (forward pass) and gradients (backward pass)
            loss_value, grads = grad(model, augmented_x, rot_label)
            # apply weight updates using the optimizer's algorithm
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            # Update progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            epoch_accuracy.update_state(rot_label, model(augmented_x, training=True))
        # epoch's end
        tr_loss.append(epoch_loss_avg.result().numpy())
        tr_acc.append(epoch_accuracy.result().numpy())

        # Compute validation loss and accuracy
     #   plot_sample(x_val[0:16], 4, 4)
        print(y_val[0:16])
        val_loss.append(get_loss(model, x_val, y_val, training=False).numpy())
        epoch_accuracy.reset_state()       # clean training accuracy data to compute val_accuracy
        epoch_accuracy.update_state(y_val, model(x_val, training=False))
        val_acc.append(epoch_accuracy.result().numpy())
        # reset states
        epoch_accuracy.reset_state()
        epoch_loss_avg.reset_state()

        # print loss
        print("\n")
        print("Epoch: {}/{},  Train_Loss: {:9.4f}   Train_acc: {:9.4f} ,   Val_Loss: {:9.4f}   Val_acc: {:9.4f} ".format(epoch+1,
              epochs, float(tr_loss[epoch]), float(tr_acc[epoch]), float(val_loss[epoch]), float(val_acc[epoch]) ))

    return ((tr_loss, val_loss), (tr_acc, val_acc))


if __name__=='__main__' :
    print(tf.config.list_physical_devices('GPU'))
    #tf.debugging.set_log_device_placement(True)
    EPOCHS = 10 # default value
    #epochs = int(sys.argv[1])

    # create the model object
    #cnn and dense options
    cnn_layers = [(96, 12),(192, 6),(192, 6),(192, 3)]
    dense_layers = [200, 200]

    rotnet = RotNet(cnn_layers, dense_layers, num_classes = 4, input_shape = (32,32,3))
    rotnet.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits = True))
    # print("Model Archtecture :")
    # print(rotnet.summary())
    # Load data
    (X, y_train), _ = load_dataset.load_data()
    # np.array to tensor
    x_train = tf.convert_to_tensor(X, dtype=tf.int32)
    x_train = tf.reshape( x_train, (-1, 32, 32, 3))
    print("Input shape : ",x_train.shape)

    # Train the model via self-supervision
    optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.1)
    history = self_supervised_train(rotnet, x_train[0:1000], EPOCHS,
                                   optimizer, batch_size=32, val_split=0.1, shuffle=True)
    plot_training_curves(history)
    #rotnet.save_weights('self_supervised.h5')
    #rotnet.load_weights('self_supervised.h5')

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
