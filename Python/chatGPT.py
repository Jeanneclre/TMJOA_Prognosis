import pandas as pd
import numpy as np
import subprocess

from sklearn.model_selection import KFold,StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import  SelectFromModel
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

import modelFunctions as mf
import Hyperparameters as hp
try:
    from xgboost import XGBClassifier
except:
    import sys
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'xgboost'])
    from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


import matplotlib.pyplot as plt


try:
    import lightgbm as lgb
except:
    import sys
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'lightgbm'])
    import lightgbm as lgb

try:
    import statsmodels.api as sm
except:
    import sys
    python = sys.executableestimator,
    subprocess.check_call([python, '-m', 'pip', 'install', 'statsmodels'])
    import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")


import os

import pickle
def choose_model(method):
    if method == "glmnet":
        # fit a logistic regression model using the glmnet package
        model = LogisticRegression()
        param_grid = hp.param_grid_lr
        
        
    elif method == "svmLinear":
        # fit a linear support vector machine model using the svm package
        model = SVC(probability=True)
        param_grid = hp.param_grid_svm
        

    elif method == "rf":
        # fit a random forest model using the random forest package
        model = RandomForestClassifier()
        param_grid = hp.param_grid_rf
        

    elif method == "xgbTree":
        # fit an XGBoost model using the xgboost package
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier()
        # model = XGBClassifier()
        param_grid = hp.param_grid_xgb
        
    elif method == "lda2":
        # fit a linear discriminant analysis model using the LDA package
        model = LinearDiscriminantAnalysis()
        param_grid = hp.param_grid_lda
        

    elif method == "nnet":
        # fit a neural network model using the MLPClassifier package
        model = MLPClassifier()
        param_grid = hp.param_grid_nnet
        

    elif method == "glmboost":
        # fit a gradient boosting model using the gbm package
        model = lgb.LGBMClassifier()
        param_grid = hp.test_param_grid_glm

    elif method == "hdda":
        # fit a high-dimensional discriminant analysis model using the pls package
        model = PCA()
        param_grid = hp.param_grid_hdda

    else:
        raise ValueError("Invalid method name. Choose from 'glmnet', 'svmLinear', 'rf', 'xgbTree', 'lda2', 'nnet', 'glmboost', 'hdda'.")
    
    return model, param_grid

def run_innerLoop(method, filename,fold):

    model,param_grid = choose_model(method)
    Nsubfolds = 10
    inner_cv = StratifiedKFold(n_splits=Nsubfolds, shuffle=True, random_state=1) # '*'
    
    predYT = []
    y_trueList = []
    scoresList = []

    # Inner Loop 
    idx = 0
    acc0 = 0
    file_toDel = 0

    for train_idx, valid_idx in inner_cv.split(X,y):
        idx +=1
        print(f'________Inner Loop {idx}_________')

        X_trainIn, X_valid= X[train_idx], X[valid_idx]
        y_trainIn, y_valid = y[train_idx], y[valid_idx]
    

        # Hyperparameter tuning with GridSearchCV if the grid is not too big, 
        # otherwise use RandomizedSearchCV - faster
        if len(param_grid) > 5:
            hyperparam_tuning = RandomizedSearchCV(estimator=model,
                                param_distributions=param_grid,
                                n_iter=10,
                                scoring='roc_auc',
                                n_jobs=-1,
                                cv=inner_cv,
                                verbose=0,
                                refit=True,
                                error_score='raise')
        else:
            hyperparam_tuning = GridSearchCV(estimator=model,
                                param_grid=param_grid,
                                scoring='roc_auc',  # use AUC 
                                n_jobs=-1,
                                cv=inner_cv,
                                verbose=0,
                                refit=True,
                                error_score='raise') 
        
       
        hyperparam_tuning.fit(X_trainIn, y_trainIn)
        best_estimator = hyperparam_tuning.best_estimator_


        # Feature selection
        try:
            print('Feature selection')
            #Meta-transformer for selecting features based on importance weights.
            sfm = SelectFromModel(best_estimator, prefit=True)
            X_train_selected = sfm.transform(X_trainIn)
            X_valid_selected = sfm.transform(X_valid)
            print('Number of features selected:', X_train_selected.shape[1])

        except:
            # If the estimator doesn't support feature selection, use all features
            print('no feature selection')
            X_train_selected = X_trainIn
            X_valid_selected = X_valid

        # ADD : update best estimator with the selected features

        # Fit the model with the best hyperparameters
        best_estimator.fit(X_train_selected, y_trainIn)

        print('best_estimator:', best_estimator)
        y_pred = best_estimator.predict(X_valid_selected)
        try:
            y_scores = best_estimator.predict_proba(X_valid_selected)[:,1]
        except:
            y_scores = ['Nan'] * len(y_pred)
            
        # compute the accuracy of the model
        accuracy = metrics.accuracy_score(y_valid, y_pred)

        # Save predictions in list
        for i in range(len(y_pred)):
            predYT.append(y_pred[i])
            y_trueList.append(y_valid[i])
            scoresList.append(y_scores[i])

        # Save the best model
        if accuracy > acc0:
            if file_toDel != 0:
                mf.delete_file(bestM_filename)
            idx_acc = idx
            acc0 = accuracy
            # Save the model into "byte stream"
            bestM_filename = filename.split('.')[0]+'_bestModel'+f'_{fold}-{idx_acc}'+'.pkl'
            pickle.dump(best_estimator, open(bestM_filename, 'wb'))
            # load the model from pickle: pickled_model = pickle.load(open('model.pkl','rb'))
            #pickled_model.predict(X_test)
            print('Best Model is number ', idx_acc)
            file_toDel = 1
        print('best model current:', bestM_filename)
    
    print('y_trueList SHAPE:', len(y_trueList))
    print('predYT SHAPE:', len(predYT))
    # Evaluation and save results in files
    accuracy,column_name,list_eval = mf.evaluation(y_trueList,predYT,scoresList,idx)
    mf.write_files(filename,column_name,list_eval)

    return predYT, y_trueList, bestM_filename



methods_list = ["glmnet", "svmLinear", "rf", "xgbTree", "lda2", "nnet", "glmboost", "hdda"]

for iii in range(0,3):
    print('**********', methods_list[iii], '**********')
    vecT = [(i, j) for i in [1, 3, 4, 5, 6, 7] for j in range(1, 9)]
    i1 = vecT[iii-1][0]
    i2 = vecT[iii-1][1]

    A = pd.read_csv("./TMJOAI_Long_040422_Norm.csv")

    y = A.iloc[:, 0].values
    X = A.iloc[:, 1:].values

    Nfold = 10
    N = 10
    seed0 = 2022

    np.random.seed(seed0)
    folds_CVT = KFold(n_splits=Nfold, shuffle=True, random_state=seed0)

    # Init files for results
    innerL_filename = f"PerformancesAUC/{methods_list[iii]}/result_{methods_list[iii]}_InnerLoop.csv"
    outerL_filename = f"PerformancesAUC/{methods_list[iii]}/result_{methods_list[iii]}_OuterLoop.csv"
    # Create the path to store the file 'result_X_InnerLoop.csv' if it doesn't exist
    if not os.path.exists(os.path.dirname(innerL_filename)):
        os.makedirs(os.path.dirname(innerL_filename))

    # Remove files if they exist
    mf.delete_file(innerL_filename)
    mf.delete_file(outerL_filename)

    y_predicts = []
    y_scroresList = []

    acc0=0
    fileToDel = 0
    for fold, (train_idx, test_idx) in enumerate(folds_CVT.split(X)):
        print(f"================Fold {fold+1}================")
    
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        predictInL, correctPred_InL, bestInnerM_filename = run_innerLoop(methods_list[iii],innerL_filename,fold+1)

        print('The best model for this fold is: ', bestInnerM_filename)
        
        # Test the best model from inner loop
        best_innerModel = pickle.load(open(bestInnerM_filename,'rb'))
        best_innerModel.fit(X_train, y_train)
        y_Fpred = best_innerModel.predict(X_test)

        # Some model doesn't have a predict_proba method
        # test if the model has a predict_proba method
        if hasattr(best_innerModel, "predict_proba"):
            y_Fscores = best_innerModel.predict_proba(X_test)[:,1]
        else:
            y_Fscores = ['Nan'] * len(y_Fpred)
        
        # Save all predictions in lists to evaluate the entire outer loop and not each fold
        for i in range(len(y_Fpred)):
            y_predicts.append(y_Fpred[i])
            y_scroresList.append(y_Fscores[i])

        acc_eachFold = metrics.accuracy_score(y_test, y_Fpred)
    
        if acc_eachFold > acc0:
            if fileToDel != 0:
                mf.delete_file(bestOuterM_filename)
                mf.delete_file(bestOuterM_eval)
            idx_BestAcc = fold+1
            acc0 = acc_eachFold
            # Save the model into "byte stream"
            bestOuterM_filename = outerL_filename.split('.')[0]+'_bestModelOuter'+f'_{fold+1}'+'.pkl'
            pickle.dump(best_innerModel, open(bestOuterM_filename, 'wb'))
            
            # evaluation of the best model
            print('pred:',y_Fpred)
            accUnused, column_name,list_eval = mf.evaluation(y_test,y_Fpred,y_Fscores,fold+1)
            bestOuterM_eval = bestOuterM_filename.split('.')[0]+'_Evaluation.csv'
            mf.write_files(bestOuterM_eval,column_name,list_eval)
            fileToDel = 1

            print('best Outer model current:', bestOuterM_filename)
    
    

    # Save predictions of the inner loop in a csv file
    prediction_filename = innerL_filename.split('.')[0]+f'_Innerpredictions{fold}'+'.csv'
    df_predict = pd.DataFrame({'Actual':correctPred_InL , 'Predicted': predictInL})
    df_predict.to_csv(prediction_filename, index=False) 

    # Evaluation of outer loop models and save results in files
    print("********Final**********")
    print('Best Model Outer Loop is number ', bestOuterM_filename)
    print('len y_predicts:', len(y_predicts))
    accuracy,column_name,list_eval = mf.evaluation(y,y_predicts,y_scroresList,fold+1)

    prediction_filename = outerL_filename.split('.')[0]+'_Finalpredictions'+'.csv'
    df_predict = pd.DataFrame({'Actual': y, 'Predicted': y_predicts})
    df_predict.to_csv(prediction_filename, index=False)
    mf.write_files(outerL_filename,column_name,list_eval)

    