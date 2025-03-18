# from re import I

import matplotlib.pyplot as plt

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

from Polynominal import *
import sys
import csv
from time import process_time
from pathlib import Path
import shutil
from modifiedOutput import *


# t1_start = process_time()


def NNfun(times,firstname,pathdir,filename_ANN,i):


    # if os.path.exists("results/NN_trained_models/models/" + firstname + ".h5"):  # or len(data[0])<3:
    #     print("NN already trained \n")
    #     print("NN loss: ", NN_eval(pathdir,filename_ANN)[0], "\n")
    #     model_pasim = NN_eval(pathdir,filename_ANN)[1]  #  return(rmse_loss(model(factors_val),product_val),  model)-->  (tensor(0.9410, grad_fn=<DivBackward0>),va khode model ham mide
    # if  os.path.exists("results/NN_trained_models/models/" + filename_ANN + ".h15"):
    #     print("Found pretrained NN \n")
    #     model_pasim = NN_train(times,pathdir,filename_ANN,pretrained_path="results/NN_trained_models/models/" + filename_ANN + ".h5")
    #     print("NN loss after training: ", NN_eval(pathdir,filename_ANN), "\n")
    # else:
    print("Training a NN on the data... \n")
    model_pasim = NN_train(times,pathdir,filename_ANN)
    print("NN loss: ", NN_eval(pathdir,filename_ANN), "\n")



    firstname= '1'


##################### Check which symmetry/separability is the best###################################
    # Symmetries


    print("Checking for symmetries...")

    symmetry_plus_result = check_translational_symmetry_plus(pathdir, filename_ANN)

    symmetry_minus_result = check_translational_symmetry_minus(pathdir, filename_ANN)

    symmetry_multiply_result = check_translational_symmetry_multiply(pathdir, filename_ANN)

    symmetry_divide_result = check_translational_symmetry_divide(pathdir, filename_ANN)


    print('Symm +', symmetry_plus_result[0:5]) #min_error, best_i, best_j, best_mu, best_sigma
    print('Symm -', symmetry_minus_result[0:5])
    print('Symm *', symmetry_multiply_result[0:5])
    print('Symm /', symmetry_divide_result[0:5])
    print("")

    print("Checking for ab**n...")
    ab_n_result = check_ab_n(pathdir, filename_ANN)
    print(ab_n_result[0:5])
    print("")


    print("Checking for a**n*cosb...")
    an_cosb_result = check_an_cosb(pathdir,filename_ANN)
    print(an_cosb_result[0:5])
    print("")

    print("Checking for an_plus_cosb...")
    an_plus_cosb_result = check_an_plus_cosb(pathdir, filename_ANN)
    print("")

    print("Checking for an_expb...")
    an_expb_result = check_an_expb(pathdir, filename_ANN)
    print("")


    print("Checking for a+b**n...")
    a_plus_bn_result = check_a_plus_b_n(pathdir,filename_ANN)
    print("")

    print("Checking for a-b**n...")
    a_minus_bn_result = check_a_minus_b_n(pathdir,filename_ANN)
    print("")




#######################################################################################

    if symmetry_plus_result[0]==-1:
        idx_min = -1
    Er=np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0],ab_n_result[0],an_cosb_result[0],an_plus_cosb_result[0],an_expb_result[0],a_plus_bn_result[0],a_minus_bn_result[0]])
    idx_min = np.argmin(Er)
    minEr=np.min(Er)
    Er_01=[symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0],ab_n_result[0],an_cosb_result[0],an_plus_cosb_result[0],an_expb_result[0],a_plus_bn_result[0],a_minus_bn_result[0]]
    Er_01.remove(np.min(Er_01))



    if times == 1:
        C = 0.8 + (i // 2) * 0.1
        # C=1

    elif times == 2:

        C = 0.6 + (i // 2) * 0.1
        # C = 1
    else:

        C = 0.3 + (i // 2) * 0.1
        # C = 1
    # C=1
    if minEr <  C*np.min(Er_01):
        idx_min = np.argmin(Er)
        print('minEr< {}*np.min(Er_01)'.format(C))
    else:
        idx_min=-1



    # idx_min = np.argmin(np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0],ab_n_result[0],an_cosb_result[0],an_plus_cosb_result[0],an_expb_result[0]  ]))# an_plus_cosb_result[0],ab_n_result[0],an_cosb_result[0],ab_n_result[0] ,an_expb_result[0],abn_1_result[0] ,,separability_plus_result[0],separability_multiply_result[0]
    print('\n')
    print(np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0] ,ab_n_result[0],an_cosb_result[0],an_plus_cosb_result[0] ,an_expb_result[0],a_plus_bn_result[0],a_minus_bn_result[0]]))#,ab_n_result[0]
    # print('idx_min=',idx_min)
    text_01=np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0] ,ab_n_result[0],an_cosb_result[0],an_plus_cosb_result[0] ,an_expb_result[0],a_plus_bn_result[0],a_minus_bn_result[0] ])
    # min_error = np.min( np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0] ])) #,ab_n_result[0],an_expb_result[0],,separability_plus_result[0],separability_multiply_result[0]
    # median_error = np.median( np.array([symmetry_plus_result[0], symmetry_minus_result[0], symmetry_multiply_result[0], symmetry_divide_result[0] ])) #,ab_n_result[0],separability_plus_result[0],separability_multiply_result[0]
    # #
    # if min_error > (0.1*median_error):
    #     print('modifying output...')
    #     mod_func(pathdir,pathdir_write_to,filename_ANN,soo,Nu)
    #     convert.convert(eq_symbols, x2, mull_coef)
    #     exit()



    if idx_min== -1:
        print('')

    if idx_min == 0:
        print("Translational symmetry '+' found for variables:", symmetry_plus_result[1],symmetry_plus_result[2])

        print("")
        new_pathdir, new_filename = do_translational_symmetry_plus(pathdir, filename_ANN,symmetry_plus_result[1],symmetry_plus_result[2],times)

    elif idx_min == 1:

        print("Translational symmetry '-' found for variables:", symmetry_minus_result[1],symmetry_minus_result[2])
        print("")

        new_pathdir, new_filename = do_translational_symmetry_minus(pathdir, filename_ANN,symmetry_minus_result[1],symmetry_minus_result[2],times)
        
    elif idx_min == 2:

        print("Translational symmetry '*' found for variables:", symmetry_multiply_result[1],symmetry_multiply_result[2])
        print("")
        new_pathdir, new_filename = do_translational_symmetry_multiply(pathdir, filename_ANN,symmetry_multiply_result[1],symmetry_multiply_result[2],times)
        
    elif idx_min == 3:

        print("Translational symmetry '/' found for variables:", symmetry_divide_result[1],symmetry_divide_result[2])
        print("")
        new_pathdir, new_filename = do_translational_symmetry_divide(pathdir, filename_ANN,symmetry_divide_result[1],symmetry_divide_result[2],times)
    #
    elif idx_min == 4:

        print("ab**n found for variables:", ab_n_result[1],ab_n_result[2])
        print("")
        new_pathdir, new_filename = do_ab_n(pathdir, filename_ANN,ab_n_result[1],ab_n_result[2],ab_n_result[-1],times)

    elif idx_min == 5:

        print("a**n*cosb found for variables:", an_cosb_result[1],an_cosb_result[2])
        print("")
        new_pathdir, new_filename = do_an_cosb(pathdir, filename_ANN,an_cosb_result[1],an_cosb_result[2], an_cosb_result[-1],times)

    elif idx_min == 6:

        print("an_plus_cosb found for variables:", an_plus_cosb_result[1],an_plus_cosb_result[2])
        print("")
        new_pathdir, new_filename = do_an_plus_cosb(pathdir, filename_ANN,an_plus_cosb_result[1],an_plus_cosb_result[2], an_plus_cosb_result[-1],times)

    elif idx_min == 7:

        print("a**n*exp(b) found for variables:", an_expb_result[1], an_expb_result[2])
        print("")
        new_pathdir, new_filename = do_an_expb(pathdir, filename_ANN, an_expb_result[1], an_expb_result[2],an_expb_result[-1],times)

    elif idx_min == 8:
        print("a+b**n found for variables:", a_plus_bn_result[1], a_plus_bn_result[2])
        print("")
        new_pathdir, new_filename = do_a_plus_b_n(pathdir, filename_ANN, a_plus_bn_result[1], a_plus_bn_result[2],a_plus_bn_result[-1],times)

    elif idx_min == 9:
        print("a-b**n found for variables:", a_minus_bn_result[1], a_minus_bn_result[2])
        print("")
        new_pathdir, new_filename = do_a_minus_b_n(pathdir, filename_ANN, a_minus_bn_result[1], a_minus_bn_result[2],a_minus_bn_result[-1],times)



    if idx_min == -1:
        besti, bestj, n_value = '', '', ''

    if idx_min == 0:
        besti, bestj,n_value,R2 = symmetry_plus_result[1],symmetry_plus_result[2],'',symmetry_plus_result[-2]

    elif idx_min == 1:
        besti, bestj,n_value,R2 = symmetry_minus_result[1],symmetry_minus_result[2] ,'',symmetry_minus_result[-2]

    elif idx_min == 2:
        besti, bestj,n_value,R2 = symmetry_multiply_result[1],symmetry_multiply_result[2] ,'',symmetry_multiply_result[-2]

    elif idx_min == 3:
        besti, bestj,n_value,R2 = symmetry_divide_result[1],symmetry_divide_result[2] ,'',symmetry_divide_result[-2]

    elif idx_min == 4:
        besti, bestj, n_value,R2 = ab_n_result[1], ab_n_result[2], ab_n_result[-1], ab_n_result[-2]

    elif idx_min == 5:
        besti, bestj, n_value,R2 = an_cosb_result[1],an_cosb_result[2],an_cosb_result[-1],an_cosb_result[-2]

    elif idx_min == 6:
        besti, bestj, n_value,R2 = an_cosb_result[1],an_cosb_result[2],an_cosb_result[-1],an_cosb_result[-2]

    elif idx_min == 7:
        besti, bestj, n_value,R2 = an_cosb_result[1], an_cosb_result[2], an_cosb_result[-1], an_cosb_result[-2]

    elif idx_min == 8:
        besti, bestj,n_value ,R2= a_plus_bn_result[1],a_plus_bn_result[2] ,a_plus_bn_result[-1],a_plus_bn_result[-2]

    elif idx_min == 9:
        besti, bestj, n_value,R2 = a_minus_bn_result[1],a_minus_bn_result[2], a_minus_bn_result[-1], a_minus_bn_result[-2]



    return new_pathdir,new_filename,besti,bestj,idx_min,n_value,minEr,Er,R2



