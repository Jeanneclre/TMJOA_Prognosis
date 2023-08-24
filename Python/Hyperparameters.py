
# dictionnary of hyperparameters according to the model

param_grid_glmnet = {
    'alpha': [1.0, 1.0],
    'lambda': [0.1, 0.01]
}

#SVM
param_grid_svmLinear = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf']
}

#Random forester
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}

#LogisticRegression
param_grid_lr = {
    'penalty': ['l1', 'l2'],         # Penalty type
    'C': [0.01, 0.1, 1.0, 10.0],     # Regularisation parameter
    'solver': ['liblinear', 'lbfgs', 'saga'],  # resolution algorithm
}
