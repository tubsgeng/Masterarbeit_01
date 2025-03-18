from os import close
import numpy as np

def convert(eq_symbols, IndenpendentVar, mull_coef,best_GP_index=0) : #,idx_min, besti, bestj ):

#open text file in read mode
    if best_GP_index==0:
        with open('Equation.txt', 'r') as f:
            lines = f.read().splitlines()
            data = lines[-1]  #last_line

            x = data.split(", ")
            gen_eq =x[0]
            #close file
            f.close()

    else:
        with open ('Poly_Equation.txt', 'r') as f:
            #lines = f.read().splitlines()
            gen_eq = f.read()
            f.close() 
    
    print(gen_eq)

    col_n=len(IndenpendentVar)+1
    new_eq = gen_eq
    if best_GP_index==0:
        x_GP =1
    else:
        x_GP=0
        col_n-=1
    # print(IndenpendentVar)
    
    for i in range (x_GP, col_n):
        x_n='x'+str(i)
        x_new='x'+"_"+str(i)
        new_eq= new_eq.replace(x_n, x_new)
        #print(new_eq)

    for i in range (x_GP, col_n):
        x_n='x'+'_'+str(i)
       # print(x_n)
        new_eq= new_eq.replace(x_n,'(%s)'%IndenpendentVar[i-1])
        #print(new_eq)
    y=eq_symbols[-1]   
    final_eq = y+'='+'(%s)'%mull_coef+'*(%s)'%new_eq
    print(final_eq)

    file_sym = open( "Final_result.txt" ,"w")  #create a file with this name and this file will be ready to write in it
    file_sym.write(str(final_eq))  
    file_sym.close()
    return final_eq

def convertNN(eq_symbols, IndenpendentVar, mull_coef):  # ,idx_min, besti, bestj ):






    y = eq_symbols[-1]
    final_eq = y + '=' + '(%s)' % mull_coef + '*(%s)' % IndenpendentVar
    print(final_eq)

    file_sym = open("Final_result.txt", "w")  # create a file with this name and this file will be ready to write in it
    file_sym.write(str(final_eq))
    file_sym.close()


def variable_name_modifier(IndenpendentVar,besti,bestj,idx_min,n_value):
    if idx_min == 0:
        IndenpendentVar[besti] ='('+ IndenpendentVar[besti]+'+'+IndenpendentVar[bestj]+')'
        del IndenpendentVar[bestj]
    elif idx_min == 1:
        IndenpendentVar[besti] ='('+ IndenpendentVar[besti]+'-'+IndenpendentVar[bestj]+')'
        del IndenpendentVar[bestj]
    elif idx_min == 2:
        IndenpendentVar[besti] = '('+IndenpendentVar[besti]+'*'+IndenpendentVar[bestj]+')'
        del IndenpendentVar[bestj] 
    elif idx_min == 3:
        IndenpendentVar[besti] = '('+IndenpendentVar[besti]+'/'+IndenpendentVar[bestj]+')'
        del IndenpendentVar[bestj] 
    elif idx_min == 4:
        IndenpendentVar[besti] ='('+ IndenpendentVar[besti]+'*'+'('+IndenpendentVar[bestj]+')'+'**'+str(n_value)+')'
        del IndenpendentVar[bestj]
    elif idx_min == 5:
        IndenpendentVar[besti] ='('+ '('+IndenpendentVar[besti]+')'+'**'+str(n_value)+'*'+'cos('+IndenpendentVar[bestj]+')'+')'
        del IndenpendentVar[bestj]
    elif idx_min == 6:
        IndenpendentVar[besti] ='('+ '('+IndenpendentVar[besti]+')'+'**'+str(n_value)+'+'+'cos('+IndenpendentVar[bestj]+')'+')'
        del IndenpendentVar[bestj] 
    elif idx_min == 7:
        IndenpendentVar[besti] ='('+ '('+IndenpendentVar[besti]+')'+'**'+str(n_value)+'*'+'exp('+IndenpendentVar[bestj]+')'+')'
        del IndenpendentVar[bestj]
    elif idx_min == 8:
        IndenpendentVar[besti] = IndenpendentVar[besti] + '+' + IndenpendentVar[bestj] +'**'+str(n_value)
        del IndenpendentVar[bestj]

    elif idx_min == 9:
        IndenpendentVar[besti] = IndenpendentVar[besti] + '-' + IndenpendentVar[bestj] +'**'+str(n_value)

        del IndenpendentVar[bestj]  


    return IndenpendentVar