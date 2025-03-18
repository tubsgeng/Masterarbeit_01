import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import autokeras as ak
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


bs = 2048
wd = 1e-2
is_cuda =0
def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def NN_train(pathdir, filename, epochs=1000, lrs=1e-2, N_red_lr=4, pretrained_path=""):
    try:
        os.mkdir("results/NN_trained_models")
    except:
        pass

    try:
        os.mkdir("results/NN_trained_models/models")
    except:
        pass
    try:
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))

        epochs = 200*n_variables

        if len(variables)<5000:
            epochs = epochs*3

        if n_variables==0 or n_variables==1:
            return 0

        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))


         # create the model
        model = Sequential()
        model.add(Dense(units=128, activation='selu', input_dim=n_variables))
        model.add(Dropout(0.2))
        model.add(Dense(units=64, activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=30, activation='selu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer= 'Adam', loss='mse', metrics=['mae','mse'])

       
         # train
        history = model.fit(variables, f_dependent, verbose=1, batch_size= bs, epochs=epochs, 
                                 validation_split=0.2)
      
            
        model.save(model.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")   
            
        #plots
        fig = plt.figure(figsize=(10,5))

        ax = fig.add_subplot(121)
        ax.plot(history.history['loss'], color='green')
        ax.set_xlabel('epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss value')

        ax = fig.add_subplot(122)
        ax.plot(history.history['val_loss'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('val_loss')
        ax.set_title('val_loss')


        fig = plt.figure(figsize=(10,5))

        ax= fig.add_subplot(121)
        ax.plot(history.history['mae'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('mae')
        ax.set_title('The metrics')

        ax= fig.add_subplot(122)
        ax.plot(history.history['val_mae'])
        ax.set_xlabel('epoch')
        ax.set_ylabel('val_mae')
        ax.set_title('The val_metrics')

        plt.show()
        
        return model 

    except NameError:
        print("Error in file: %s" %filename)
        raise



# for run this file :

pathdir = 'C:/Users/somayeh/Desktop/feynmanpratctice/Somaye_Amin/Final_V2.2/'
filename ='Random_formula_data.txt'  # I create more data here instead of 200 , i created 1000 datapoint to see the evaluation rate of model

model_pasim, loss = NN_train(pathdir, filename, epochs=1000 , lrs=1e-2, N_red_lr=4, pretrained_path="")
print(model_pasim, loss)
#epochs=1000 , lrs=1e-2, N_red_lr=4