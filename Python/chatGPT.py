import pandas as pd
import numpy as np
import subprocess
import random
from sklearn.model_selection import KFold, train_test_split,StratifiedKFold, GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression,ElasticNet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

# try:
#     from glmnet import LogitNet
# except:
#     import sys
#     python = sys.executable
#     subprocess.check_call([python, 'pip', 'install', 'glmnet'])
#     from glmnet import LogitNet
try: 
    from scipy.sparse import csc_matrix
except:
    import sys
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'scipy'])
    from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
try:
    import seaborn as sns
except:
    import sys
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', 'seaborn'])
    import seaborn as sns

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

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import SelectFromModel

import os

import pickle
def choose_model(method):
    if method == "glmnet":
        # fit a logistic regression model using the glmnet package
        # model = ElasticNet()
        model = LogisticRegression()
        param_grid = hp.param_grid_lr
        # param_grid = hp.param_grid_glmnet
        
        
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
        param_grid = hp.param_grid_glm

    elif method == "hdda":
        # fit a high-dimensional discriminant analysis model using the pls package
        model = sm.multiclass.HDDA()
        param_grid = hp.param_grid_hdda

    else:
        raise ValueError("Invalid method name. Choose from 'glmnet', 'svmLinear', 'rf', 'xgbTree', 'lda2', 'nnet', 'glmboost', 'hdda'.")
    
    return model, param_grid

def run_innerLoop(method, filename,fold):

    model,param_grid = choose_model(method)
    Nsubfolds = 10
    inner_cv = StratifiedKFold(n_splits=Nsubfolds, shuffle=True, random_state=1)
    
    predYT_acc = []
    predYT = []
    y_trueList = []
    scoresList = []

    # Inner Loop - Hyperparameter Tuning
    idx = 0
    acc0 = 0
    file_toDel = 0

    for train_idx, valid_idx in inner_cv.split(X,y):
        # print('train_idx subfold:', train_idx)
        # print('valid_idx subfold:', valid_idx)
        idx +=1
        print(f'________Inner Loop {idx}_________')

        X_trainIn, X_valid= X[train_idx], X[valid_idx]
        y_trainIn, y_valid = y[train_idx], y[valid_idx]

        hyperparam_tuning = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='accuracy',
                            n_jobs=-1,
                            cv=inner_cv,
                            verbose=0,
                            refit=True,
                            error_score='raise')
        
        hyperparam_tuning.fit(X_trainIn, y_trainIn)
        best_estimator = hyperparam_tuning.best_estimator_


        # Feature selection
        # Check if the estimator supports feature selection
        # For classification estimators with 'coef_' attribute (e.g., LinearSVC, LogisticRegression)
        try:
            print('Feature selection')
            sfm = SelectFromModel(best_estimator, prefit=True)
            X_train_selected = sfm.transform(X_trainIn)
            X_valid_selected = sfm.transform(X_valid)

        except:
            # If the estimator doesn't support feature selection, use all features
            print('no feature selection')
            X_train_selected = X_trainIn
            X_valid_selected = X_valid


        best_estimator.fit(X_train_selected, y_trainIn)

        print('best_estimator:', best_estimator)
        y_pred = best_estimator.predict(X_valid_selected)
        try:
            y_scores = best_estimator.predict_proba(X_valid_selected)[:,1]
            scoresList.append(y_scores.tolist())
        except:
            y_scores = None
            scoresList = None

        

        # compute the accuracy of the model
        accuracy = metrics.accuracy_score(y_valid, y_pred)
        predYT_acc.append(accuracy)

        # Save predictions in list
        for i in range(len(y_pred)):
            
            predYT.append(y_pred[i])
            y_trueList.append(y_valid[i])
    
        

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
        print('best model actually:', bestM_filename)
    
    print('type(predYT):', type(predYT))
    print('type y_trueList:', type(y_trueList))
    accuracy,column_name,list_eval = mf.evaluation(y_trueList,predYT,scoresList,idx)
    mf.write_files(filename,column_name,list_eval)

 
    prediction_filename = filename.split('.')[0]+'_Innerpredictions'+'.csv'
    df_predict = pd.DataFrame({'Actual': y_trueList, 'Predicted': predYT})
    df_predict.to_csv(prediction_filename, index=False)

    return predYT, predYT_acc, bestM_filename


iii = 3
methods_list = ["glmnet", "svmLinear", "rf", "xgbTree", "lda2", "nnet", "glmboost", "hdda"]
vecT = [(i, j) for i in [1, 3, 4, 5, 6, 7] for j in range(1, 9)]
i1 = vecT[iii-1][0]
i2 = vecT[iii-1][1]

A = pd.read_csv("./TMJOAI_Long_040422_Norm.csv", skiprows=[0])

y = A.iloc[:, 0].values
X = A.iloc[:, 1:].values

Nfold = 10
N = 10
seed0 = 2022

np.random.seed(seed0)
folds_CVT = KFold(n_splits=Nfold, shuffle=True, random_state=seed0)

# Init files for results
innerL_filename = f"ChatGPT_r/{methods_list[iii]}/result_{methods_list[iii]}_InnerLoop.csv"
outerL_filename = f"ChatGPT_r/{methods_list[iii]}/result_{methods_list[iii]}_OuterLoop.csv"
# Create the path to store the file 'result_X_InnerLoop.csv' if it doesn't exist
if not os.path.exists(os.path.dirname(innerL_filename)):
    os.makedirs(os.path.dirname(innerL_filename))

# Remove files if they exist
mf.delete_file(innerL_filename)
mf.delete_file(outerL_filename)


Shap = np.zeros((len(y), X.shape[1]))
colnames_Shap = list(A.columns)[1:]
select = np.empty(N)

y_predicts = []

for fold, (train_idx, test_idx) in enumerate(folds_CVT.split(X)):
    print(f"================Fold {fold+1}================")
   
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    

    predict_HTFS, accuracy, bestInnerM_filename = run_innerLoop(methods_list[iii],innerL_filename,fold+1)

    print('The best model for this fold is: ', bestInnerM_filename)
    
    # Test the best model from inner loop
    best_innerModel = pickle.load(open(bestInnerM_filename,'rb'))
    best_innerModel.fit(X_train, y_train)
    y_Fpred = best_innerModel.predict(X_test)
    try:
        y_Fscores = best_innerModel.predict_proba(X_test)[:,1]
    except:
        y_Fscores = None
    
    # Save all predictions in list
    for i in range(len(y_Fpred)):
        y_predicts.append(y_Fpred[i])
    

print("********Final**********")
accuracy,column_name,list_eval = mf.evaluation(y,y_predicts,y_Fscores,fold+1)

prediction_filename = outerL_filename.split('.')[0]+'_Finalpredictions'+'.csv'
df_predict = pd.DataFrame({'Actual': y, 'Predicted': y_predicts})
df_predict.to_csv(prediction_filename, index=False)
mf.write_files(outerL_filename,column_name,list_eval)

    