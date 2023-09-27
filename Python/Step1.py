
# Functions used for splitting the data in folds, hyperparameters tuning, feature selection, evaluation, etc.
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import  SelectFromModel
from sklearn import metrics

# Import models from sklearn and from other packages
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.mixture import GaussianMixture # HDDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier


try:
    from xgboost import XGBClassifier
except:
    import sys
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'xgboost'])
    from xgboost import XGBClassifier

# Import other files of the project
import modelFunctions as mf
import Hyperparameters as hp

# Useful libraries
import os
import pickle
import pandas as pd
import numpy as np
import subprocess

def choose_model(method):
    """
    Give the right model and the right hyperparameters grid for the method
    Input: method: name of the method
    Output: model: model from the method list, param_grid: hyperparameters grid
    """
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
        model = XGBClassifier()
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
        model = GradientBoostingClassifier()
        param_grid = hp.param_grid_glmboost


    elif method == "hdda":
        # fit a high-dimensional discriminant analysis model using the pls package
        model = GaussianMixture(n_components=2)
        param_grid = hp.param_grid_hdda
    else:
        raise ValueError("Invalid method name. Choose from 'glmnet', 'svmLinear', 'rf', 'xgbTree', 'lda2', 'nnet', 'glmboost', 'hdda'.")
    
    return model, param_grid

def runFS(model,X_train,X_valid,y_train,y_valid):
    """
    Features selection -> Best features selected depending on a number  "nb_features"
    Input: model: model from the method list, X_train: Data, y_train: Target, y_valid: Correct answer
    Output: X_train_selected: Data with selected features, X_valid_selected: Data with selected features
    """
    auc0 =0
    nb_features = [3,5,10,15,20]
    for nb in nb_features:
        
        # If the model doesn't have a coef_ attribute, use feature_importances_ attribute
        if  hasattr(model, "feature_importances_"):
            Coefficient = np.abs(model.feature_importances_)
        elif hasattr(model, "coef_"):
            Coefficient=np.abs(model.coef_)[0]
        else:
            
            Coefficient = np.sum(np.abs(model.coefs_[0]), axis=1)
        
        sorted_indices = np.argsort(Coefficient)
        top_N_largest_indices = sorted_indices[-nb:]
        X_train_selected = X_train[:,top_N_largest_indices]
        X_valid_selected = X_valid[:,top_N_largest_indices]

        #Evaluate the model with the selected features 
        # To choose the right number of features to get the best AUC
        model.fit(X_train_selected, y_train)
        y_scores = model.predict_proba(X_valid_selected)[:,1]
        y_scores = np.array(y_scores).astype(float)
        auc = metrics.roc_auc_score(y_valid,y_scores)
        if auc > auc0 :
            auc0 = auc
            nb_selected_features = nb
            X_trainSelected_final = X_train_selected
            X_validSelected_final = X_valid_selected
    
    return nb_selected_features,X_trainSelected_final,X_validSelected_final
    # try:
    #     print('Feature selection')
    #     #Meta-transformer for selecting features based on importance weights.
    #     sfm = SelectFromModel(model,threshold=0.004)
    #     X_train_selected = sfm.transform(X_train)
    #     X_valid_selected = sfm.transform(X_valid)

    #     sfm.fit(X_train_selected, y_train)
    #     print('***Number of features selected: ', X_train_selected.shape)


    # except:
    #     # If the estimator doesn't support feature selection, use all features
    #     print('no feature selection')
    #     X_train_selected = X_train
    #     X_valid_selected = X_valid
           
    # return X_train_selected, X_valid_selected

def hyperparam_tuning(model, param_grid, X_train, y_train, inner_cv):
    # Hyperparameter tuning with GridSearchCV if the grid is not too big, 
    # otherwise use RandomizedSearchCV - faster
    if model == 'GaussianMixture()':
        scorer = metrics.make_scorer(metrics.accuracy_score)
    else:
        scorer = metrics.make_scorer(metrics.roc_auc_score)

    if len(param_grid) > 5:
        hyperparam_tuning = RandomizedSearchCV(estimator=model,
                            param_distributions=param_grid,
                            n_iter=10,
                            scoring=scorer,
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise')
    else:
        hyperparam_tuning = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring=scorer,  
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise') 
        
    hyperparam_tuning.fit(X_train, y_train)
    best_estimator = hyperparam_tuning.best_estimator_
    return best_estimator

def run_innerLoop(methodFS,methodPM, filename,X,y ,fold):

    modelFS,param_gridFS = choose_model(methodFS)
    modelPM,param_gridPM = choose_model(methodPM)
    Nsubfolds = 10
    inner_cv = StratifiedKFold(n_splits=Nsubfolds, shuffle=True, random_state=1) # should I put a seed here too (if yes, the same as in outer_cv)?
    
    predYT = []
    y_trueList = []
    scoresList = []

    # Inner Loop 
    idx = 0
    acc0 = 0
    file_toDel = 0
    bestM_filename = filename.split('.')[0]+'_bestModel'+'_NULL'+'.pkl'
    for train_idx, valid_idx in inner_cv.split(X,y):
        idx +=1
        import time
        print(f'________Inner Loop {idx}_________')

        # print the day and hour
        print('run began at',time.localtime()) 

        X_train, X_valid= X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
    
        ## Feature selection ##
        # 1st: hyperparameter tuning for the FS model
        best_estimatorFS= hyperparam_tuning(modelFS, param_gridFS, X_train, y_train, inner_cv)
        # 2nd: run the FS model with the best hyperparameters
        NbFeatures, X_train_selected,X_valid_selected= runFS(best_estimatorFS,X_train, X_valid,y_train,y_valid)

        # Hyperparameter tuning for the predictive model (PM)
        best_estimator = hyperparam_tuning(modelPM, param_gridPM, X_train_selected, y_train, inner_cv)
        print('best_estimator for PM:', best_estimator)

        best_estimator.fit(X_train_selected, y_train)
        
        y_pred = best_estimator.predict(X_valid_selected)
      
        y_scores = best_estimator.predict_proba(X_valid_selected)[:,1]
        
        y_scores = np.array(y_scores).astype(float) 
        # compute the AUC of the model
        auc = metrics.roc_auc_score(y_valid,y_scores)
        

        # Save predictions in list
        for i in range(len(y_pred)):
            predYT.append(y_pred[i])
            y_trueList.append(y_valid[i])
            scoresList.append(y_scores[i])

        # Save the best model according to the AUC
        if auc > acc0 :
            idx_acc = idx
            acc0 = auc
            # Save the model into "byte stream"
            bestM_filename = filename.split('.')[0]+'_bestModel'+f'_{fold}-{idx_acc}'+'.pkl'
            if file_toDel != 0:
                mf.delete_file(bestM_filename)

            pickle.dump(best_estimator, open(bestM_filename, 'wb'))
            # load the model from pickle: pickled_model = pickle.load(open('model.pkl','rb'))
            #pickled_model.predict(X_test)
            print('Best Model is number ', idx_acc)
            file_toDel = 1
        print('best model current:', bestM_filename)

    # Evaluation and save results in files
    # filename = "TestThresh/thresholdsEval-06.csv"
    column_name,list_eval = mf.evaluation(y_trueList,predYT,scoresList,idx)[1:]
    column_name.insert(0,'Nb Features Selected')
    list_eval.insert(0,NbFeatures)
    mf.write_files(filename,column_name,list_eval)

    return predYT, y_trueList, bestM_filename,NbFeatures


def OuterLoop(X, y,methodFS, methodPM, innerL_filename, outerL_filename):
    """
    Outer loop of the nested cross validation
    Input: X: Data, y: Target, innerL_filename: Name of the file for inner loop results, 
    outerL_filename: Name of the file for outer loop results, methods_list: List of the methods, iii: index of the method
    Output: "<outerL_filename>.csv" file
    """
    print('********Outer Loop**********')
    
    Nfold = 10
    N = 10
    seed0 = 2022

    np.random.seed(seed0)
    folds_CVT = StratifiedKFold(n_splits=Nfold, shuffle=True, random_state=seed0)

    y_predicts = []
    y_scroresList = []


    acc0=0
    fileToDel = 0
    for fold, (train_idx, test_idx) in enumerate(folds_CVT.split(X,y)):
        print(f"================Fold {fold+1}================")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        predictInL, correctPred_InL, bestInnerM_filename, NbFeatures = run_innerLoop(methodFS,methodPM, innerL_filename,X_train,y_train,fold+1)

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

        #Keep the best model with the best AUC
        auc = metrics.roc_auc_score(y_test,y_Fscores)
        if auc > acc0:
            if fileToDel != 0:
                mf.delete_file(bestOuterM_filename)
                mf.delete_file(bestOuterM_eval)
            idx_BestAcc = fold+1
            acc0 = auc
            # Save the model into "byte stream"
            bestOuterM_filename = outerL_filename.split('.')[0]+'_bestModelOuter'+f'_{fold+1}'+'.pkl'
            pickle.dump(best_innerModel, open(bestOuterM_filename, 'wb'))
            
            # evaluation of the best model
            column_name,list_eval = mf.evaluation(y_test,y_Fpred,y_Fscores,fold+1)[1:]
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
    column_name,list_eval = mf.evaluation(y,y_predicts,y_scroresList,fold+1)[1:]
    column_name.insert(0,'Nb Features Selected')
    list_eval.insert(0,NbFeatures)
    mf.write_files(outerL_filename,column_name,list_eval)


    #Save evaluation in a csv file for each model that is runned in outerloop and add to the previous data
    # Add in the first index the name of the model
    list_eval.insert(0,f'{methodPM}_{methodFS}')
    column_name.insert(0,'Model PM_FS')
    performance_filename = 'Final_Performances48-FunctFS.csv'
    mf.save_performance(list_eval, column_name, performance_filename)


    prediction_filename = outerL_filename.split('.')[0]+'_Finalpredictions'+'.csv'
    df_predict = pd.DataFrame({'Actual': y, 'Predicted': y_predicts})
    df_predict.to_csv(prediction_filename, index=False)
    