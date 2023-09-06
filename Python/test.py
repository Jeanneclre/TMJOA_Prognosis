import subprocess
import sys
import os
import numpy as np
import pandas as pd
import pickle

try :
    import sklearn
except :
    print("imblearn not found")
    python = sys.executable
    subprocess.check_call([python,'pip', 'install', 'sklearn'], stdout=subprocess.DEVNULL)
    import sklearn

from sklearn import metrics as mt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Import different models:
from sklearn.linear_model import LogisticRegression


# Import other scripts 
import Hyperparameters as hp
import modelFunctions as mf

# Import Input Data (csv file)
inputFilename = "./TMJOAI_Long_040422_Norm.csv"
df = pd.read_csv(inputFilename,skiprows=[0])

first_column = mf.convertToArray(df.iloc[:, 0])  
other_columns = mf.convertToArray(df.iloc[:, 1:])

y= first_column
X= other_columns
# Initialize the model
model = LogisticRegression()

# pipeline 
pipe = Pipeline([('std', StandardScaler()),
                  ('lr', model)])

# setting param grid
param_grid = hp.param_grid_lr
param_grid1 = [{'lr__penalty': ['l2','l1'],
                'lr__C': np.power(10., np.arange(-4, 4))}]

# Nested CV with parameter optimization
outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
idx = 0
for train, test in outer_cv.split(X,y):
    idx+=1
    # Use the training set in this fold of the outer cv to find the best
    # hyperparameters. This loop will find the best hyperparameters for the
    # model and then retrain the model on the entire training set.
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    clf = GridSearchCV(estimator=pipe, param_grid=param_grid1, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    clf.fit(X[train], y[train])
    

    df_result = pd.DataFrame(clf.cv_results_)
    df_result = df_result.sort_values(by=['rank_test_score'])
    df_result.to_csv(f'test/result{idx}.csv', index=False) # QUE 8 LIGNES DE RESULTATS!!!

    # Test the model on the held aside testing set from the outer cv
    y_pred = clf.best_estimator_.predict(X[test])
    print(f"----------------PERFOMANCES FOLD {idx}-----------------")
    print("accuracy:", mt.accuracy_score(y[test],y_pred))
    print("ROC AUC:",mt.roc_auc_score(y[test],y_pred))
    print("Best Parameters from grid search: ", clf.best_params_)
    print("Best Score from grid search: ", clf.best_score_)
    print("Best Estimator from grid search: ", clf.best_estimator_)
    print("--------------------------------------------")
    