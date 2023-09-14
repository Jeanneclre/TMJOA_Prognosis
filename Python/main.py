
# This file is used to run the entire project
#################################################
##        Contributor: Jeanne CLARET           ##
##                                             ##
##            TMJOA_PROGNOSIS                  ##
##                                             ##
#################################################


iii=0 # First index in python is 0


method_list = {
    "lasso": "Step1_Glmnet.py",
    "ridge": "Step1_ridge.py",
    "svc": "Step1_SVC.py",
    "random_forest": "Step1_RF.py",
    "xgboost": "Step1_XGboost.py", 
    "lda": "Step1_LDA.py", 
    "neural_network": "Step1_NNET.py", 
    "gradient_boosting": "Step1_GBM.py"
}


Nfeature_selection = [1, 3, 4, 5, 6, 7]
Npredictive_modeling = list(range(1, 9)) 

# Generate every possible combination of the two lists
vecT = [(val1, val2) for val2 in Npredictive_modeling for val1 in Nfeature_selection ]

i1 = vecT[iii][0]
i2 = vecT[iii][1]