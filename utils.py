import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json

def plot_sample(X, rows, cols, tensor=False):
    ''' Function for plotting images.'''

    nb_rows = rows
    nb_cols = cols
    fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(8, 8))
    k=0
    # if input is tensor convert to np.array before plotting
    if tensor: X = X.numpy()
    for i in range(0, nb_rows):
        for j in range(0, nb_cols):
            axs[i, j].xaxis.set_ticklabels([])
            axs[i, j].yaxis.set_ticklabels([])
            axs[i, j].imshow(X[k])
            plt.tight_layout()
            k += 1
    plt.show()


def plot_training_curves(history):
    '''Plot training learning curves for both train and validation.'''
    #Defining the metrics we will plot.
    train_acc = history[1][0]
    val_acc = history[1][1]
    train_loss = history[0][0]
    val_loss = history[0][1]

    #Range for the X axis.
    epochs = range(len(train_loss))

    #Plotting Loss figures.
    fig = plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.rcParams.update({'font.size': 22}) #configuring font size.
    plt.plot(epochs,train_loss,c="red",label="Training Loss") #plotting
    plt.plot(epochs,val_loss,c="blue",label="Validation Loss")
    plt.xlabel("Epochs") #title for x axis
    plt.ylabel("Loss")   #title for y axis
    plt.legend(fontsize=11)

    #Plotting Accuracy figures.
    fig = plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.plot(epochs,train_acc,c="red",label="Training Acc") #plotting
    plt.plot(epochs,val_acc,c="blue",label="Validation Acc")
    plt.xlabel("Epochs")   #title for x axis
    plt.ylabel("Accuracy") #title for y axis
    plt.legend(fontsize=11)

    plt.show()


class job_receiver:
    '''
    Basic job receiver class. Receives a json file and
    produces the relevant dictionary...
    '''

    def __init__(self, path: str):
        self.path = path

    def __call__(self):
        with open(self.path, 'r') as file:
            job = json.load(file)
        return job

if __name__ == '__main__':

    #run this script to understand the implemented functionality...

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    #rotnet example...

    #path for rotnet construction and training...
    rotnet_path = './rotnet_config_example.json'

    #rotnet job (dictionary form)
    rotnet_job = job_receiver(rotnet_path)()

    #here the rotnet model is constructed. see the json file in the rotnet_path to understand...
    RotNet = RotNet_constructor(rotnet_job['build_instructions'])

    print(RotNet.summary())

    '''
    Here we imagine training to happen...
    Options from rotnet_job['training']
    '''

    #save model...
    tf.keras.models.save_model(RotNet, rotnet_job['save_path'])

    ###########################################################################################
    ###########################################################################################
    ###########################################################################################

    #prednet example...

    #path for prednet construction and training...
    prednet_path = './prednet_config_example.json'

    #rotnet job (dictionary form)
    prednet_job = job_receiver(prednet_path)()

    #here the rotnet model is constructed. see the json file in the rotnet_path to understand...
    PredNet = PredNet_constructor(prednet_job['build_instructions'])

    print(PredNet.summary())

    '''
    Here we imagine training to happen...
    Options from prednet_job['training']
    '''

    #save model...
    PredNet.save_weights(prednet_job['save_path'])
