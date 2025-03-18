from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
from torch.utils import data
import pickle
from matplotlib import pyplot as plt
import torch.utils.data as utils
import time
import os


bs = 2048
wd = 1e-2

is_cuda = torch.cuda.is_available()
is_cuda =0

def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))
    return torch.sqrt(F.mse_loss(pred, targ))/denom

def NN_train(times,pathdir, filename, epochs=1000, lrs=1e-2, N_red_lr=4, pretrained_path=""):
    print('times=',times)
    try:
        os.mkdir("results/NN_trained_models")
    except:
        pass

    try:
        os.mkdir("results/NN_trained_models/models")
    except:
        pass
    try:
        print(pathdir)
        print(filename)
        n_variables = np.loadtxt(pathdir+"%s" %filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+"%s" %filename, usecols=(0,))

        epochs = 100*n_variables

        if len(variables)<5:
            epochs = epochs*3

        if n_variables==0 or n_variables==1:
            return 0

        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+"%s" %filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir+"%s" %filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables)
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()

        product = torch.from_numpy(f_dependent)
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        class SimpleNet(nn.Module):
            def __init__(self, ni):
                super().__init__()
                # self.linear1 = nn.Linear(ni, 128),nn.BatchNorm1d(128, eps=1e-05)
                self.linear1 = nn.Linear(ni, 128)
                self.linear2 = nn.Linear(128, 128)
                self.linear3 = nn.Linear(128, 128)
                self.linear4 = nn.Linear(128, 64)
                self.linear5 = nn.Linear(64, 64)
                self.linear6 = nn.Linear(64, 1)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                x = torch.relu(self.linear3(x))
                x = torch.relu(self.linear4(x))
                x = torch.relu(self.linear5(x))
                x = self.linear6(x)
                return x

            # def forward(self, x):
            #     x = torch.tanh(self.linear1(x))
            #     x = torch.tanh(self.linear2(x))
            #     x = torch.tanh(self.linear3(x))
            #     x = torch.tanh(self.linear4(x))
            #     x = self.linear5(x)
            #     return x

        my_dataset = utils.TensorDataset(factors,product) # create your datset be jaye daden vorudi khoruji be kelas dataset MulDataset 
        my_dataloader = utils.DataLoader(my_dataset, batch_size=bs, shuffle=True) # create your dataloader --> dataloader az in dataset data barmidare mide be gpu
         # dataloader az dataset dade migire mide be gpu --> pas bayad begim az kodum dataset bardar bede be gpu , az tarafi dade ha ra bach bach mide be gpu pas behesh batch size ro ham midahim 
         # stochastic gradient descent vs bach size gradient vs mini batch 
        if is_cuda:
            model_pasim = SimpleNet(n_variables).cuda()
        else:
            model_pasim  = SimpleNet(n_variables)

        if pretrained_path!="":
            model_pasim .load_state_dict(torch.load(pretrained_path)) # # get the tunned weights

        check_es_loss = 10000

        for i_i in range(N_red_lr):
            optimizer_pasim  = optim.Adam(model_pasim .parameters(), lr = lrs)
            for epoch in range(epochs):  # epochs for morure dade ha 
                model_pasim .train()
                for i, data in enumerate(my_dataloader): # numerate: value ha ba index shan ra midahad 
                    optimizer_pasim .zero_grad()
                
                    if is_cuda:
                        fct = data[0].float().cuda()
                        prd = data[1].float().cuda()
                    else:
                        fct = data[0].float()
                        prd = data[1].float()
                    
                    loss = rmse_loss(model_pasim (fct),prd)
                    loss.backward()
                    optimizer_pasim.step()
                
                '''
                # Early stopping
                if epoch%20==0 and epoch>0:
                    if check_es_loss < loss:
                        break
                    else:
                        torch.save(model_feynman.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")
                        check_es_loss = loss
                if epoch==0:
                    if check_es_loss < loss:
                        torch.save(model_feynman.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")
                        check_es_loss = loss
                '''
                torch.save(model_pasim.state_dict(), "results/NN_trained_models/models/" + filename + ".h5")   
            lrs = lrs/10

        return model_pasim, loss 

    except NameError:
        print("Error in file: %s" %filename)
        raise



# for run this file :

# pathdir = 'C:/Users/somayeh/Desktop/feynmanpratctice/Somaye_Amin/Final_V2.2/'
# filename ='Random_formula_data.txt'  # I create more data here instead of 200 , i created 1000 datapoint to see the evaluation rate of model

# model_pasim, loss = NN_train(pathdir, filename , lrs=1e-2, N_red_lr=4, pretrained_path="")
# print(model_pasim, loss)  # ---------> finla loss =  tensor(0.0018, grad_fn=<DivBackward0>)
# # #epochs=1000 , lrs=1e-2, N_red_lr=4