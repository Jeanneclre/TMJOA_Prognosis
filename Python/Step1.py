import csv

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
# Import different models:
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC #estimator = SVC(kernel='linear', probability=True)  # kernel='linear' pour un SVM linéaire
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn import metrics as mt

import numpy as np

import Hyperparameters as hp


def convertToArray(list1):
    arr = np.array(list1)
    return arr

def get_score(model, X_test, y_test):
    return model.score(X_test, y_test)

iii=0 # First index in python is 0

method_list = ["lasso", "ridge", "svc", "random_forest", "xgboost", "lda", "neural_network", "gradient_boosting"]


Nfeature_selection = [1, 3, 4, 5, 6, 7]
Npredictive_modeling = list(range(1, 9)) 

# Generate every possible combination of the two lists
vecT = [(val1, val2) for val2 in Npredictive_modeling for val1 in Nfeature_selection ]

i1 = vecT[iii][0]
i2 = vecT[iii][1]


# Input
inputFilename = "./TMJOAI_Long_040422_Norm.csv"

# Read in the data from the CSV file
with open(inputFilename, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # Skip the header row
    # Initialize the lists to store the data
    first_column = []
    other_columns = []

    for row in csv_reader:
        
        # Store the first column in the first list
        first_column.append(row[0])
        
        # Store the other columns in the second list
        other_columns.append(row[1:])

#print("other_columns: ", other_columns) #np.shape to get (h,w) of the matrix

NbFold = 10
seed0=2022
np.random.seed(seed0)


skf = StratifiedKFold(n_splits=NbFold, shuffle=True, random_state=seed0)

# stores folds for cross-validation
foldsCVT = list(skf.split(other_columns,first_column))

auc_scorer = mt.make_scorer(mt.mean_squared_error, greater_is_better=True)
# param_grid_simple = {
#     'n_estimators': [100],
#     'max_depth': [None]
# }

# train_control = GridSearchCV(
#     estimator=RandomForestClassifier(), 
#     param_grid=param_grid_simple, 
#     scoring='accuracy',
#     cv=foldsCVT,
#     verbose=1,
#     refit=True
# )

train_control = RandomForestClassifier()

# Split the data into folds and separate the training data from the validation data
for train_idx, valid_idx in skf.split(other_columns, first_column):
    # Training data
    X_train, y_train = [other_columns[i] for i in train_idx], [first_column[i] for i in train_idx]
    # Validation data
    X_valid, y_valid = [other_columns[i] for i in valid_idx], [first_column[i] for i in valid_idx]

    # Convert to numpy arrays
    X_train = convertToArray(X_train)
    y_train = convertToArray(y_train)
    X_valid = convertToArray(X_valid)
    y_valid = convertToArray(y_valid)

    # Train the model
    train_control.fit(X_train, y_train)

    # Predict on the validation data
    y_pred = train_control.predict(X_valid)

    mse = mt.mean_squared_error(y_valid.astype(int), y_pred.astype(int))
    score = get_score(train_control, X_valid, y_valid)
    print("score: ", score)





y = first_column
X = other_columns

for fold in range(NbFold) :
    print("fold: ", fold)

    # indtempT = foldsCVT[fold+40]
    # print("indtempT: ", indtempT)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)