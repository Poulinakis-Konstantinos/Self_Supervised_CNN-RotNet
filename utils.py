import matplotlib.pyplot as plt 

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