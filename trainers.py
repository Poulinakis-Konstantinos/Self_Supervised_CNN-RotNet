import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy.ndimage import rotate

def rotate_image(images):
    ''' Rotate the given image by 0, 90, 180, 270 degrees
    images(tensor) : A tensor of shape (num_imgs, num_rows, num_columns, num_channels) '''
    rotated_imgs = []
    for image in images:
        rotated_imgs.append(image)
        rotated_imgs.append(rotate(image, 90))
        rotated_imgs.append(rotate(image, 180))
        rotated_imgs.append(rotate(image, 270))
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
    if y == None:
        X_train, X_test = train_test_split(X.numpy(),  **options)
        X_train, X_test = tf.constant(X_train), tf.constant(X_test)
        return X_train, X_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(),  **options)
        X_train, X_test = tf.constant(X_train), tf.constant(X_test)
        y_train, y_test = tf.constant(y_train), tf.constant(y_test)
        return X_train, X_test, y_train, y_test


def get_loss(model, x, y, training=True,
             f_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    ''' Calculate loss value between ground truth y and model's output, on input x .
        If x is not a probability distribution use from_logits=True. '''

    y_ = model(x, training=training)
    return f_loss(y_true=y, y_pred=y_)


def grad(model, inputs, targets, f_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    ''' Compute the gradient of the loss in respect to the model's learnable parameters. '''
    with tf.GradientTape() as tape:
        l_value = get_loss(model, inputs, targets,
                           training=True, f_loss=f_loss)  # compute loss
    # compute gradients
    grads = tape.gradient(l_value, model.trainable_variables)
    return l_value, grads


def self_supervised_trainer(model, x, epochs, optimizer, batch_size=32, val_x=None, val_y=None, val_split=0.2, shuffle=True, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)):
    ''' Custom training loop for self-supervised learning with image rotations '''
    # lists to store values for visualization
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []

    if val_x == None and val_y == None:
        # create validation split
        x_train, x_val = train_test_split_tensors(
            x, y=None, test_size=val_split, shuffle=shuffle)
    else:
        x_train = x
        x_val = val_x
    # create the augmented validation data with their respective rotation labels
    x_val, y_val = rotate_image(x_val)

    print("====="*10)
    print(f"Initializing Self-Supervised Training for {epochs} epochs")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in tqdm(range(epochs)):
        # Training loop - using batches of batch_size
        for index, offset in enumerate(range(0, x_train.shape[0], batch_size)):
            # avoid out of bounds error while batching
            if (offset + batch_size < x_train.shape[0]):
                upper = offset + batch_size
            else:
                upper = x_train.shape[0]

            # creating batch of size batch_size
            x_batch = x_train[offset: upper]
            # Create rotated images and rotation labels
            augmented_x, rot_label = rotate_image(x_batch)

            # calculate loss (forward pass) and gradients (backward pass)
            loss_value, grads = grad(model, augmented_x, rot_label)
            # apply weight updates using the optimizer's algorithm
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update progress, add current batch loss, accuracy
            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(rot_label, model(augmented_x, training=True))

        # epoch's end. Append loss for plotting
        tr_loss.append(epoch_loss_avg.result().numpy())
        tr_acc.append(epoch_accuracy.result().numpy())

        # Compute Validation loss and accuracy
        val_loss.append(get_loss(model, x_val, y_val, training=False).numpy())
        epoch_accuracy.reset_state()       # clean accuracy state to compute val_accuracy
        epoch_accuracy.update_state(y_val, model(x_val, training=False))
        val_acc.append(epoch_accuracy.result().numpy())

        # reset states
        epoch_accuracy.reset_state()
        epoch_loss_avg.reset_state()

        # Print Epoch's progress details
        print("\n")
        print("Epoch: {}/{},  Train_Loss: {:9.4f}   Train_acc: {:9.4f} ,   Val_Loss: {:9.4f}   Val_acc: {:9.4f} ".format(epoch+1,
              epochs, float(tr_loss[epoch]), float(tr_acc[epoch]), float(val_loss[epoch]), float(val_acc[epoch])))

    # return training history
    return ((tr_loss, val_loss), (tr_acc, val_acc))


def supervised_trainer(model, x, y, epochs, optimizer, batch_size=32, val_x=None, val_y=None, val_split=0.2, shuffle=True):
    ''' Custom training loop for supervised learning '''
    # lists to store values for visualization
    tr_loss = []
    val_loss = []
    tr_acc = []
    val_acc = []

    if val_x == None and val_y == None:
        # create validation split
        x_train, x_val, y_train, y_val = train_test_split_tensors(x, tf.convert_to_tensor(y, tf.int32),
                                                                  test_size=val_split, shuffle=shuffle)
    else:
        x_train = x
        y_train = y
        x_val = val_x
        y_val = val_y

    print(f"Initializing Training for {epochs} epochs")
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for epoch in tqdm(range(epochs)):
        # Training loop - using batches of batch_size
        for index, offset in enumerate(range(0, x_train.shape[0], batch_size)):
            # avoid out of bounds error while batching
            if (offset + batch_size < x_train.shape[0]):
                upper = offset + batch_size
            else:
                upper = x_train.shape[0]

            # creating batch of size batch_size
            x_batch = x_train[offset: upper]
            y_batch = y_train[offset: upper]

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

        # Compute Validation loss and accuracy
        val_loss.append(get_loss(model, x_val, y_val, training=False).numpy())
        epoch_accuracy.reset_state()       # clean accuracy state to compute val_accuracy
        epoch_accuracy.update_state(y_val, model(x_val, training=False))
        val_acc.append(epoch_accuracy.result().numpy())

        # reset states
        epoch_accuracy.reset_state()
        epoch_loss_avg.reset_state()

        # Print Epoch's progress details
        print("\n")
        print("Epoch: {}/{},  Train_Loss: {:9.4f}   Train_acc: {:9.4f} ,   Val_Loss: {:9.4f}   Val_acc: {:9.4f} ".format(epoch,
              epochs, float(tr_loss[epoch]), float(tr_acc[epoch]), float(val_loss[epoch]), float(val_acc[epoch])))

    # return training history
    return ((tr_loss, val_loss), (tr_acc, val_acc))
