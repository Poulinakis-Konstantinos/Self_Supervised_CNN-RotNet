from loaders import load_data #Img_dloader
from RotNet import RotNet, RotNet_constructor, eval_rotnet
from PredNet import PredNet_constructor
from trainers import self_supervised_trainer, supervised_trainer
from utils import plot_sample, plot_training_curves, job_receiver

from os import path, environ, makedirs
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import  Adadelta,Adam
import numpy as np
import sys 
import json

SAVE_PATH = 'Saved_models'
CONSTRUCTOR_PATH = 'constructors'
RESULTS_PATH = 'results'
test_size = 10

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()
    #samples = x_train[0:4]
    #plot_sample(samples, 2, 2)

    # Check execution device
    print(tf.config.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)

    # Convert data to tensors of float32 type
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)

    # #################### METHOD 1 : Creating Model from Class ####################
    # #Model Hyperparameters
    # cnn_layers = [(96, 12), (192, 6), (192, 6), (192, 3)]
    # dense_layers = [200, 200]
    # num_classes = 4
    # IMG_SHAPE = [32, 32, 3]
    # # Training Hyperparameters
    # EPOCHS = 1  # default value
    # LR = 0.1
    # BATCH_SIZE = 32
    # VAL_SPLIT = 0.1
    # optimizer = Adadelta(learning_rate=LR)
    # loss_function = SparseCategoricalCrossentropy(from_logits=True)

    # # create RotNet Model (self-supervised)
    # rotnet = RotNet(cnn_layers, dense_layers, num_classes=num_classes, input_size=IMG_SHAPE)

    # history = self_supervised_trainer(rotnet, x_train[0:10], EPOCHS,
    #                                   optimizer, batch_size=BATCH_SIZE,
    #                                   val_split=VAL_SPLIT, shuffle=True)
    # print(rotnet.summary())
    # plot_training_curves(history)
    # # Evaluate the RotNet based on rotation prediction
    # rotnet.eval_on_rotations(x_test[0:10])

    
    ################### METHOD 2 : Creating Model from Constructor ##################
    #path for rotnet construction and training...     
    if len(sys.argv) > 1 :
        rotnet_path = path.join(CONSTRUCTOR_PATH, sys.argv[1])
    else : 
        rotnet_path = path.join(CONSTRUCTOR_PATH, './rotnet_config_example.json')
    #rotnet job (dictionary form)
    rotnet_job = job_receiver(rotnet_path)()

    # path for prednet construction and training...
    if len(sys.argv) > 2 :
        prednet_path = path.join(CONSTRUCTOR_PATH, sys.argv[2])
    else : 
        prednet_path = path.join(CONSTRUCTOR_PATH, './prednet_config_example.json')
    # rotnet job (dictionary form)
    prednet_job = job_receiver(prednet_path)()

    environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    #here the rotnet model is constructed. see the json file in the rotnet_path to understand...
    rotnet = RotNet_constructor(rotnet_job['build_instructions'])
    print(rotnet.summary())

    if (prednet_job['build_instructions']["transfer"]):
        train_par = rotnet_job["training"]
        optimizer = Adadelta(learning_rate=train_par['learning_rate'])
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        history = self_supervised_trainer(rotnet,
                                          x_train[:train_par["dataset_size"]],
                                          train_par['epochs'],
                                          optimizer,
                                          batch_size=train_par['batch_size'],
                                          val_split=train_par['val_split'],
                                          shuffle=train_par['shuffle'],loss=loss)

        result_path = path.join(RESULTS_PATH, rotnet_job['build_instructions']['name'])
        if not path.exists(result_path) : makedirs(result_path)
        eval_rotnet(x_test=x_test[:test_size], model=rotnet, result_path=result_path)
        plot_training_curves(history,  result_path)
        with open(path.join(result_path, "architecture.txt"), 'w') as f:
            rotnet.summary(print_fn=lambda x: f.write(x + '\n'))
        with open(path.join(result_path, "configurations.json"), 'w') as f:
            #f.write(str(train_par))
            f.write(json.dumps(train_par)) # use `json.loads` to do the reverse
           # print(train_par , file=f)


    #save model...
    print("Saving RotNet model at ", path.join(SAVE_PATH, rotnet_job["save_path"]))
    # Save the entire RotNet model
    rotnet.save(path.join(SAVE_PATH, rotnet_job["save_path"]))

    #######   Initialize PredNet  #########
    print('Initializing PredNet Network')
    # here the rotnet model is constructed. see the json file in the rotnet_path to understand...
    prednet = PredNet_constructor(prednet_job['build_instructions'])
    print(prednet.summary())

    train_par = prednet_job["training"]
    optimizer = Adadelta(learning_rate=train_par['learning_rate'])
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    history = supervised_trainer(prednet,
                                 x_train[0:train_par["dataset_size"]],
                                 y_train[0:train_par["dataset_size"]],
                                 train_par['epochs'],
                                 optimizer,train_par['batch_size'],
                                 None,None,
                                 val_split=train_par['val_split'],
                                 shuffle=train_par['shuffle'],loss=loss)
   
    result_path = path.join(RESULTS_PATH, prednet_job['build_instructions']['name'])
    if not path.exists(result_path) : makedirs(result_path)
    plot_training_curves(history, result_path) 
    # Testing
    evaluations = prednet.evaluate(x_test[:test_size], y_test[:test_size], batch_size = 32)
    print("====="*4 + "  PredNet evaluation Report  " + "====="*4)
    print(f"Loss = {evaluations[0]}")
    print(f"Accuracy = {evaluations[1]}")
    print("======="*10)
    with open(path.join(result_path,'evaluation_result.txt'), 'w') as txt :
        txt.write(f" Model : {result_path.split(path.sep)[1]} \n Test set accuracy is : {evaluations[1]} \n Test set Loss is : {evaluations[0]}")
    with open(path.join(result_path, "architecture.txt"), 'w') as f:
        prednet.summary(print_fn=lambda x: f.write(x + '\n'))
    with open(path.join(result_path, "configurations.json"), 'w') as f:
        #txt.write(str(train_par) + f'\n Self-Supervision : {prednet_job["build_instructions"]["transfer"]}' )
        f.write(json.dumps(train_par)) # use `json.loads` to do the reverse
        #print(train_par + f'\n Self-Supervision : {prednet_job["build_instructions"]["transfer"]}' , file=txt)
        
    pred = prednet.predict(x_test[:test_size])
    prednet.save(path.join(SAVE_PATH, prednet_job["save_path"]))

    print('END')