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
from torch.optim.lr_scheduler import CosineAnnealingLR
from matplotlib import pyplot as plt
import time
from sklearn.metrics import r2_score
is_cuda = torch.cuda.is_available()
is_cuda =0
bs = 2048


def rmse_loss(pred, targ):
    denom = targ**2
    denom = torch.sqrt(denom.sum()/len(denom))

    return torch.sqrt(F.mse_loss(pred, targ))/denom

def R2(pred, targ):

    return 1 - torch.sum((pred - targ) ** 2) / torch.sum( (targ - torch.mean(targ)) ** 2)


def NN_eval(pathdir,filename):
    try:
        n_variables = np.loadtxt(pathdir+filename, dtype='str').shape[1]-1
        variables = np.loadtxt(pathdir+filename, usecols=(0,))

        if n_variables==0:
            return 0
        elif n_variables==1:
            variables = np.reshape(variables,(len(variables),1))
        else:
            for j in range(1,n_variables):
                v = np.loadtxt(pathdir+filename, usecols=(j,))
                variables = np.column_stack((variables,v))

        f_dependent = np.loadtxt(pathdir+filename, usecols=(n_variables,))
        f_dependent = np.reshape(f_dependent,(len(f_dependent),1))

        factors = torch.from_numpy(variables[0:int(5*len(variables)/6)]) # a percentage of all variables are selected here 
        if is_cuda:
            factors = factors.cuda()
        else:
            factors = factors
        factors = factors.float()
        product = torch.from_numpy(f_dependent[0:int(5*len(f_dependent)/6)])
        if is_cuda:
            product = product.cuda()
        else:
            product = product
        product = product.float()

        factors_val = torch.from_numpy(variables[int(5*len(variables)/6):int(len(variables))]) # baghimandeye variable ha inja hastand
        if is_cuda:
            factors_val = factors_val.cuda()
        else:
            factors_val = factors_val
        factors_val = factors_val.float()
        product_val = torch.from_numpy(f_dependent[int(5*len(variables)/6):int(len(variables))])      
        if is_cuda:
            product_val = product_val.cuda()
        else:
            product_val = product_val
        product_val = product_val.float()

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

        if is_cuda:
            model = SimpleNet(n_variables).cuda()
        else:
            model = SimpleNet(n_variables)
                    
        model.load_state_dict(torch.load("results/NN_trained_models/models/"+filename+".h5"))
        model.eval()
        return(rmse_loss(model(factors_val),product_val),model)
        # return (R2(model(factors_val), product_val), model)

    except Exception as e:
        print(e)
        return (100,0)



# inorder to run this file :
# pathdir = 'C:/Users/somayeh/Desktop/feynmanpratctice/Somaye_Amin/Final_V2.2/'
# filename ='Random_formula_data.txt'  # I create more data here instead of 200 , i created 1000 datapoint to see the evaluation rate of model

# model_pasim, loss = NN_eval(pathdir, filename)
# print(model_pasim, loss)


