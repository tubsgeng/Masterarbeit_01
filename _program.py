from re import I
import copy
import atexit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import DimensionalAnalysis1
import GeneticAlgorith
import os
import convert
from NN_train import NN_train
from NN_train import NN_train
from NN_eval import NN_eval
#from S_symmetry import *
from Basic_Operations import *
from Trigonometric_Operations import*
from Exponential_operations import *
from S_separability import *
from sklearn.preprocessing import minmax_scale
from Polynominal import *
import sys
import csv
from time import process_time
from pathlib import Path
import shutil
from modifiedOutput import *
import  time
import sys
from  NNfunc import *

import warnings
warnings.filterwarnings("ignore")
import sys
import copy


print("Program statt!")

Nu =614
# Select a formula number
# Here it is recommended to run the equation 21 in the extrusion process, it can show all the functions more completely



t1_start =time.time()

eq = open('Data/vars/extrusion/var{}.txt'.format(Nu), 'r')

eq_data = eq.read()

eq_symbols = eq_data.split(',')


print('eq_symbols=', eq_symbols)

filename = 'Data/equations/extrusion/eq{}.txt'.format(Nu)

pathdir = '//Users/gengliu/Downloads/Yibo-Code/'    # Please change to the path of your own file

pathdir_write_to = '/Users/gengliu/Downloads/Yibo-Code/Modified_Output/'



try:
    os.remove("NewVariables{}.txt".format(Nu))
except:
    pass

try:
    os.remove("DataNewVariables{}.txt".format(Nu))
except:
    pass

try:
    os.remove("EX41_50.txt_dim_red_variables{}.txt".format(Nu))
except:
    pass

# try:
#     os.remove("Equation{}.txt".format(Nu))
# except:
#     pass

try:

    os.remove(filename + '.csv')
except:
    pass
try:
    os.remove("Final_result.txt")
except:
    pass



DA = DimensionalAnalysis1.dimensionalAnalysis(pathdir, filename, eq_symbols, Nu)


if DA == 0:

    D_file = open('NewVariables{}.txt'.format(Nu), 'r')
    data1 = D_file.read()
    IndenpendentVar = data1.split(',')
    mull_coef = IndenpendentVar[1]
    DA_data = np.loadtxt('DataNewVariables{}.txt'.format(Nu))

    final_eq = eq_symbols[-1] + '=' + str(DA_data[0]) + mull_coef
    file_sym = open("Final_result{}.txt".format(Nu), "w")
    file_sym.write(str(final_eq))
    print(final_eq)
    file_sym.close()
    t1_stop = time.time()
    try_time = t1_stop - t1_start
    print('try time=', try_time)
    print('DimensionalAnalysis solved the equation')
    exit()

else:
    print("DA can't solve the equation ")
    print(' ')



filename_ANN = 'DataNewVariables{}.txt'.format(Nu)
D_file = open('NewVariables{}.txt'.format(Nu), 'r')
data1 = D_file.read()
IndenpendentVar = data1.split(',')
mull_coef = IndenpendentVar[1]
print('mull_coef=', mull_coef)


del IndenpendentVar[0:2]
del IndenpendentVar[-1]

l=copy.deepcopy(IndenpendentVar)  # Make a deep copy of the initial Indenpendent variables to prevent the index from changing when it is operated later

print('IndenpendentVar=', IndenpendentVar)  # Difference from the original program is the variable names have been changed

firstname = filename_ANN

maxdeg = 3
best_GP = 0
best_GP_index = 0
n_value = 0
print('checking Genetic algorithm...')
print("")
soo = GeneticAlgorith.genetic(pathdir, filename_ANN,Nu)  # Running GP for the first time
so=soo[0]
best_GP = so[1]
setofdata=soo[2]


if best_GP> 0.9999:
    print('the score of GP learn is good enough: '+ 'R2 score=', '%.6f'%so[1])
    final_eq=convert.convert(eq_symbols, IndenpendentVar, mull_coef,best_GP_index)
    t1_stop=time.time()
    try_time = t1_stop - t1_start
    print('try time=', try_time)
    exit()                       #  End the program if the result is good enough

print('\n')
print('the score of GP learn is not good enough: '+'\n'+'R2 score=', '%.6f'%so[1])
print('Result of Gp:')
final_eq=convert.convert(eq_symbols, IndenpendentVar, mull_coef, best_GP_index)
print("")


with open(pathdir+filename_ANN) as f:
    reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
    first_row = next(reader)
    num_cols = len(first_row)
if num_cols<3:                               # Run mod_fuc when 'Just one variable!'
    print("Just one variable!")
    print("")
    print('modifying output...')
    R2modfunc=mod_func(pathdir,pathdir_write_to,filename_ANN,soo,Nu)
    final_eq=convert.convert(eq_symbols, IndenpendentVar, mull_coef, best_GP_index=0)
    print('R2 atfer modfuc:','%.6f'%R2modfunc)
    t1_stop = time.time()
    try_time = t1_stop - t1_start
    print('try time=', try_time)
    exit()

else:
    print('Try Neural Networks')       # When the first GP result is not good enough , try NN




R2N = []  #Create a container to keep R2 after each step of NN

VarN=[]   #Create a container to keep Variables after each step of NN

def NNF(i,pathdir,filename_ANN,times,IndenpendentVar):

        while(times<len(l)-1)  :  # 'l' from the previous deep copy

            print('IndenpendentVar',IndenpendentVar)

            times += 1

            new_pathdir, new_filename, besti, bestj, idx_min, n_value, minEr, Er, R2 = NNfun(times, firstname, pathdir, filename_ANN, i)
            pathdir, filename_ANN = new_pathdir, new_filename

            R2N.append(R2)

            VarNN = convert.variable_name_modifier(IndenpendentVar, besti, bestj, idx_min, n_value)

            VarN.append(str(VarNN))

            print('idx_min=', idx_min)

        return VarNN,minEr,Er,new_pathdir,new_filename,R2,R2N




times = 0    # The number of training times of the neural network, each training step, the number of times plus 1

i=0

# the whole ' try ->  except -> else ' structure is to allow the neural network to cycle multiple times to prevent misjudgment caused by its own errors.

while i<10:          # 'i' is the maximum number of runs of the neural network

    i+=1

    try:
        VarNN,minEr,Er,new_pathdir,new_filename,R2ofNN,R2N=NNF(i,pathdir,filename_ANN,times,IndenpendentVar)

    except:                                # When the error value is not small enough, retrain
        print("Error not good enough")
        IndenpendentVar = data1.split(',')
        del IndenpendentVar[0:2]
        del IndenpendentVar[-1]
        print('i=', i)
        R2N=[]
        VarN = []
    else:                                  # When the error value is good enough, stop

        print('\n')
        print('VarNN=', VarNN)
        print('Result of NN:')
        final_eq=convert.convertNN(eq_symbols, VarNN, mull_coef)
        t1_stop = time.time()
        try_time = t1_stop - t1_start
        print('try time=', try_time)
        break




# When the result of the neural network is not good enough, take the R2 optimal independent variable in the intermediate step as input of the second GP
R2ofNN = 0  # 添加这行代码来定义变量R2ofNN

if R2ofNN!=1:
    print('\n')
    print('Result not good enough,truncate the optimal variable and reuse GP :')

    print('Var in NN',VarN)   # Independent variables for all intermediate steps

    VarforGP2= list(eval(VarN[int(np.argmax(R2N))]))  # Take out the best Independent variables


    bp=int(np.argmax(R2N))+1

    print('best R2 index:',bp)

    print('IndependentVar in GP2',VarforGP2)

    soo = GeneticAlgorith.genetic2(new_pathdir='results/forGP2/', new_filename='times'+str(bp)) # Rerun GP
    so=soo[0]

    print('final R2 score:','%.6f'%so[1])
    best_GP_index=0
    final_eq=convert.convert(eq_symbols, VarforGP2, mull_coef, best_GP_index)
    t1_stop = time.time()
    try_time = t1_stop - t1_start
    print('try time=', try_time)
    print('\n')

finalR2score=so[1]
# print(finalR2score)
# Run modfunc when the result is not good enough
if finalR2score != 1 :
    print('Result still not good enough, try mod_func')
    R2modfunc=mod_func(pathdir=pathdir,pathdir_write_to=pathdir_write_to,filename=filename_ANN,so=soo,Nu=Nu)
    print('R2 after mod func:','%.6f'%R2modfunc)

    if  R2modfunc < finalR2score:    # If the result of modfunc is not good enough, output the result of the second GP
        print('\n')
        print('final_equal:')
        final_eq = convert.convert(eq_symbols, VarforGP2, mull_coef, best_GP_index)
        t1_stop = time.time()
        try_time = t1_stop - t1_start
        print('try time=', try_time)



