import os
import sklearn.metrics as mt
import numpy as np
import csv



def convertToArray(list1):
    arr = np.array(list1)
    return arr

def evaluation(y_true,y_pred,y_scores,idx):
    """
    Evaluation of the model
    Input: y_true: Correct answer, y_pred: Predictions from the model, y_scores: Probability of the predictions
    Output: int(Accuracy) + list useful to create the csv file
    """

    #confusion matrix with sklearn
    
    confusMat= mt.confusion_matrix(y_true,y_pred)
    # Check the shape to make sure it's a 2x2 matrix
    if confusMat.shape != (2, 2):
        if y_true[0]==0:
            # Manually construct a 2x2 matrix
            confusMat = np.array([[confusMat[0][0], 0], [0, 0]])
        else:
            confusMat = np.array([[0, 0], [0, confusMat[0][0]]])
    
    tn,fp,fn,tp= confusMat.ravel()

  
    errors = fp+fn
    accuracy = round(mt.accuracy_score(y_true,y_pred)*100,3)
    precision = round(mt.precision_score(y_true,y_pred,zero_division=np.nan),3)
    recall = round(mt.recall_score(y_true,y_pred),3)
    specificity = round(tn/(tn+fp),3)
    f1 = round(mt.f1_score(y_true,y_pred),3)

    try:
        print('AUC is calculated')
        auc = round(mt.roc_auc_score(y_true,y_scores),3)
    except :
        auc = 'NaN'

   
    print('-----evaluation-----')
    print('Number of errors: ', errors)
    print(f'Accuracy: {accuracy} %')
    print(f'Precision score tp/(tp+fp) : {precision} ') #best value is 1 - worst 0
    print(f'Recall score tp/(fn+tp): {recall} ') # Interpretatiom: High recall score => model good at identifying positive examples
    print(f'Specificity tn/(tn+fp): {specificity} ')
    # Most important scores:
    print(f'F1 : {f1} ') 
    print(f'AUC : {auc} ') # best is 1


    column_name = ['Total','Nb Errors', 'Accuracy', 'Precision', 'Recall','Specificity','F1', 'AUC']
    list_eval = [len(y_pred),errors, accuracy, precision, recall, specificity, f1, auc]


    return accuracy,column_name,list_eval

def write_files(filename,listHeader,listParam):
    """
    Write the results in a csv file
    Input: filename: Name of the file, listHeader: List of the header, listParam: List of the parameters
    Output: "<filename>.csv" file
    """
    existing_file = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        if not existing_file:
            csvwriter.writerow(listHeader)

        csvwriter.writerow(listParam)

import pandas as pd
import os

def save_performance(list_eval, column_name, filename):
    """
    Function to create a csv file with the header column_name and the row list_eval
    If the file has already 49 rows, it is deleted and a new empty file is created
    It is used to save the performance of the model
    Input: list_eval: list of the performance, column_name: list of the header, filename: name of the file
    Output: "<filename>.csv" file
    """
    # Check if the file already exists
    if os.path.isfile(filename):
        # If it does, read it
        df = pd.read_csv(filename)
        
        # Check if the file has 49 rows
        if len(df) == 49:
            # If it does, delete the file
            os.remove(filename)
            
            # And create a new empty file with the same name
            df = pd.DataFrame(columns=column_name)
            df.to_csv(filename, index=False)
    else:
        # If the file does not exist, create a new DataFrame with column names and save it
        df = pd.DataFrame(columns=column_name)
        df.to_csv(filename, index=False)
    
    # Convert the list_eval to a DataFrame row
    new_row = pd.DataFrame([list_eval], columns=column_name)
    new_row.to_csv(filename, mode='a', header=False, index=False)



def delete_file(filename) -> None:
    """ 
    Remove file named filename if it exist
    Input: Name of the file
    """
    if os.path.exists(filename) :
        os.remove(filename)


################ Old functions ################

def remove_files(filename,NbFold=10, idx_kept=None):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(filename) and i !=idx_kept:
            os.remove(filename)

def remove_files_results(NbFold=10, Outputpath='out/'):
    """ 
    Remove files if they exist
    """
    for i in range(NbFold):
        if os.path.exists(f'{Outputpath}predict{i}.csv') :
            os.remove(f'{Outputpath}predict{i}.csv')
        if os.path.exists(f'{Outputpath}result{i}.csv') :
            os.remove(f'{Outputpath}result{i}.csv')