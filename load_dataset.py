import tensorflow as tf 

def load_data() :
    ''' load Cifar10 dataset''' 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    return  (x_train, y_train), (x_test, y_test)

