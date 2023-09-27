
# This file is used to run the entire project
#################################################
##        Contributor: Jeanne CLARET           ##
##                                             ##
##            TMJOA_PROGNOSIS                  ##
##                                             ##
#################################################

import pandas as pd
import os

import modelFunctions as mf
import Step1 as st1

methods_list = ["glmnet", "svmLinear", "rf", "xgbTree", "lda2", "nnet", "glmboost", "hdda"]

vecT = [(i, j) for i in range(0, 8) for j in [0, 2, 3, 4, 5, 6] ]
print('vecT:',vecT)
A = pd.read_csv("./TMJOAI_Long_040422_Norm.csv")

y = A.iloc[:, 0].values
print('len(y):',len(y))
X = A.iloc[:, 1:].values
print('len(X):',len(X))

for iii in range(0,6):
    i_PM = vecT[iii][0]

    i_FS = vecT[iii][1]
    if i_FS == 3 or i_FS == 4 or i_FS == 5 :
        print('Xgbtree avoid')
        iii +=1
        i_PM = vecT[iii][0]
        i_FS = vecT[iii][1]
  

    print(f'====== FS with {methods_list[i_FS]} ======')
    print(f'________ Model train - {methods_list[i_PM]} ________')

    # Init files for results
    innerL_filename = f"PerformancesAUC/{methods_list[i_PM]}_{methods_list[i_FS]}/scores_{methods_list[i_PM]}_InnerLoop.csv"
    outerL_filename = f"PerformancesAUC/{methods_list[i_PM]}_{methods_list[i_FS]}/result_{methods_list[i_PM]}_OuterLoop.csv"
    if not os.path.exists(os.path.dirname(innerL_filename)):
        os.makedirs(os.path.dirname(innerL_filename))

    # Remove files if they exist
    mf.delete_file(innerL_filename)
    mf.delete_file(outerL_filename)

    
    st1.OuterLoop(X, y, methods_list[i_FS], methods_list[i_PM], innerL_filename, outerL_filename)