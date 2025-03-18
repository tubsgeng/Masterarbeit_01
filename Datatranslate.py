from __future__ import print_function
import torch
import os
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
from S_remove_input_neuron import remove_input_neuron
import time
from sklearn.preprocessing import minmax_scale

is_cuda = torch.cuda.is_available()
is_cuda=0

def trans(a):


    train_daten1 = pd.read_csv('Ex{}S.csv'.format(a), sep=' ', engine='python')
    column_headers = list(train_daten1.columns.values)
    print(column_headers)
    # np.savetxt("results/translated_data_minus/"+file_name , data_translated)
    np.savetxt('EqData/extrusion/EX{}_Symb.txt'.format(a), column_headers,delimiter=',',fmt = '%s')

    num_col = len(column_headers)
    train_daten = np.c_[pd.read_csv('EX{}S.csv'.format(a), sep=' ', engine='python')]
    np.savetxt('EqData/extrusion/EX{}.txt'.format(a), train_daten,delimiter=',',fmt = '%s')
    print(train_daten)

