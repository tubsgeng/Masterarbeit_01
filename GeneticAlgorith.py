import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from scipy import *
from sympy import *
from sklearn.model_selection import train_test_split
from gplearn.functions import make_function



def genetic(pathdir, filename,Nu):
    print(filename)
    # data = np.loadtxt(pathdir+filename)
    data = np.loadtxt("DataNewVariables{}.txt".format(Nu)) #,usecols=(0, 1), skiprows=1) # array
    df1 = np.savetxt('DataNewVariables{}.csv'.format(Nu), data, delimiter = ",")
    df = pd.read_csv('DataNewVariables{}.csv'.format(Nu), dtype=np.float32)
    # print('df in ga',df)
    setofdata = df.shape[0] + 1
    label=[]
    for i in range (len(df.columns)):
        label.append('x'+str(i+1))
    

    df.columns = label

    y = df.iloc[:,-1]
    x = df.iloc[:,0:-1]

    print(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        shuffle=True
        )


    converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'sin': lambda x    : sin(x),
    'cos': lambda x    : cos(x),
    'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'pow3': lambda x: x**3,
    'pow2': lambda x: x**2,
     'plus':lambda x:x+1,
    'minus':lambda x:x-1,
    'protected_exponent':lambda x:exp**x,}

        

    def plus(x):
        f = x+1
        return f

    plus = make_function(function=plus,name='plus',arity=1)

    def minus(x):
        f = x-1
        return f
    minus = make_function(function=minus,name='minus',arity=1)

    def protected_exponent(x):
        with np.errstate(over='ignore'):
         return np.where(np.abs(x) < 20, np.exp(x), 0.)

    exp = make_function(function=protected_exponent,name='exp',arity=1)

    # add the new function to the function_set
    population_size=1000
    # population_size = 10000
    generations=5
    # generations = 15
    function_set = ['add', 'sub', 'mul', 'div','cos','inv','sqrt', plus, minus] #,,exp,'cos','sin','sqrt','neg','inv','log', plus, minus
    gp_model = SymbolicRegressor(population_size=population_size,function_set=function_set,init_method='full',
                            generations=generations, stopping_criteria=0.01,
                            p_crossover=0.6, p_subtree_mutation=0.1,
                            p_hoist_mutation=0.05, p_point_mutation=0.2,
                            max_samples=0.9, verbose=1,
                            parsimony_coefficient=0.01, random_state=0,
                            feature_names=x_train.columns) #half and half

    gp_model.fit(x_train, y_train)

    print(gp_model._program)
    # Import sympy
    next_e = sympify((gp_model._program), locals=converter)
    print(next_e)
    print('population_size=',population_size)
    print('generations=', generations)
    so = ('R2:',gp_model.score(x_test,y_test))
    with open( "Equation.txt" ,"a") as file_sym:  #creat a file with this name and this file will be ready to write in it
        #eq_last = file_sym.read()
        #file_sym.write(eq_last)
        file_sym.write('\n')
        file_sym.write(str(next_e))                                   # write this into the file 
        file_sym.write(", ")
        file_sym.write(str(so)) 



    file_sym.close()
    return so, next_e,setofdata


def genetic2(new_pathdir, new_filename):
    print(new_filename)
    # data = np.loadtxt(pathdir+filename)
    data = np.loadtxt(new_pathdir+new_filename)  # ,usecols=(0, 1), skiprows=1) # array
    df1 = np.savetxt(new_pathdir+new_filename+'GP2', data, delimiter=",")
    df = pd.read_csv(new_pathdir+new_filename+'GP2', dtype=np.float32)

    label = []
    for i in range(len(df.columns)):
        label.append('x' + str(i + 1))

    df.columns = label

    y = df.iloc[:, -1]
    x = df.iloc[:, 0:-1]
    print(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        shuffle=True
    )

    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'sin': lambda x: sin(x),
        'cos': lambda x: cos(x),
        'inv': lambda x: 1 / x,
        'sqrt': lambda x: x ** 0.5,
        'pow3': lambda x: x ** 3,
        'pow2': lambda x: x ** 2,
        'plus': lambda x: x + 1,
        'minus': lambda x: x - 1,
        'protected_exponent': lambda x: exp ** x, }

    def plus(x):
        f = x + 1
        return f

    plus = make_function(function=plus, name='plus', arity=1)

    def minus(x):
        f = x - 1
        return f

    minus = make_function(function=minus, name='minus', arity=1)

    def protected_exponent(x):
        with np.errstate(over='ignore'):
            return np.where(np.abs(x) < 20, np.exp(x), 0.)

    exp = make_function(function=protected_exponent, name='exp', arity=1)

    # add the new function to the function_set
    # First Test
    function_set = ['add', 'sub', 'mul', 'div', 'neg', 'inv', 'log', plus,
                    minus]  # ,,exp,'cos','sin','sqrt','neg','inv','log', plus, minus
    population_size = 1000
    generations = 5
    print('population_size=', population_size)
    print('generations=', generations)
    gp_model = SymbolicRegressor(population_size=population_size, function_set=function_set, init_method='full',
                                 generations=generations, stopping_criteria=0.01,
                                 p_crossover=0.6, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.2,
                                 max_samples=0.9, verbose=1,
                                 parsimony_coefficient=0.01, random_state=0,
                                 feature_names=x_train.columns)  # half and half

    gp_model.fit(x_train, y_train)

    print(gp_model._program)
    # Import sympy
    next_e = sympify((gp_model._program), locals=converter)

    so = ('R2:', gp_model.score(x_test, y_test))
    with open("Equation.txt",
              "a") as file_sym:  # creat a file with this name and this file will be ready to write in it
        # eq_last = file_sym.read()
        # file_sym.write(eq_last)
        file_sym.write('\n')
        file_sym.write(str(next_e))  # write this into the file
        file_sym.write(", ")
        file_sym.write(str(so))

    file_sym.close()
    return so, next_e