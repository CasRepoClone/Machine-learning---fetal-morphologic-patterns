#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# please start venv before running this script and installing the required packages
#
# .\.venv\Scripts\activate
#
"""                                  Created on the 14th/05/2025
                                            Caspian North 
    identifying the type of fetal morphologic pattern based on cardiotocogram (CTG) diagnostic features."""
# data labels 
# LB     AC   FM     UC     DL   DS   DP  ASTV  MSTV  ALTV  ...   Min    Max  Nmax  Nzeros   Mode   Mean  Median  Variance  Tendency  CLASS
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import time 
from colorama import init, Fore, Back, Style

# -- reporting function --
def report(y_val, y_pred):
    print("\n")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred, average='macro', zero_division=0))
    print("Recall:", recall_score(y_val, y_pred, average='macro', zero_division=0))
    print("F1:", f1_score(y_val, y_pred, average='macro', zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
    print("\n")

def run(training, validation, hyperparameters, standard=None):
    
    # Create a Normalizer, scaler and Naive Bayes instance
    normalizer = Normalizer()
    standadizer = preprocessing.StandardScaler()
    model = GaussianNB()
    tree = DecisionTreeClassifier(
        max_depth=hyperparameters['max_depth'],                  # limits the depth of the tree
        min_samples_split=hyperparameters['min_samples_split'],  # minimum number of samples to split a node
        criterion=hyperparameters['criterion']                   # 'entropy' or 'gini'
    )
    # drop the CLASS column from the training and validation data
    X_train = training.drop(columns=['CLASS'])
    X_val = validation.drop(columns=['CLASS'])

    # set our target variable for our classifier 
    y_train, y_val = training['CLASS'], validation['CLASS']

    # create a normalized and standardized version of the training data
    X_train_normalized = normalizer.fit_transform(X_train)
    X_train_standardized = standadizer.fit_transform(X_train)

    # Transform the validation data
    X_val_normalized = normalizer.transform(X_val)
    X_val_standardized = standadizer.transform(X_val)

    # Train the model on training data for both normalizer and standardizer on nieve bayes and decision tree
    # if standadizer is not None then we will use the standardized data

    if(standard): # naive bayes data
        model.fit(X_train_standardized, y_train)
    else:model.fit(X_train_normalized, y_train)

    if(standard): # tree data 
        tree.fit(X_train_standardized, y_train)
    else:tree.fit(X_train_normalized, y_train)

    print("Naive Bayes Classifier")
    if standard: # predict on the standardized data or normalized data for naive bayes
        y_pred = model.predict(X_val_standardized) 
    else:y_pred = model.predict(X_val_normalized)

    print("Naive Bayes Accuracy:", accuracy_score(y_val, y_pred))
    report(y_val, y_pred)

    print("Decision Tree Classifier")
    if standard: # predict on the standardized data or normalized data for decision tree
        y_pred = tree.predict(X_val_standardized) 
    else:y_pred = tree.predict(X_val_normalized)

    print("Decision tree Accuracy:", accuracy_score(y_val, y_pred))
    report(y_val, y_pred)
    return model, tree, normalizer, standadizer # return all models and normalizer and standardizer

logo1 = """
██████╗  ██████╗ ██╗     ██╗████████╗███████╗ ██████╗██╗  ██╗███╗   ██╗██╗██╗  ██╗ █████╗     
██╔══██╗██╔═══██╗██║     ██║╚══██╔══╝██╔════╝██╔════╝██║  ██║████╗  ██║██║██║ ██╔╝██╔══██╗    
██████╔╝██║   ██║██║     ██║   ██║   █████╗  ██║     ███████║██╔██╗ ██║██║█████╔╝ ███████║    
██╔═══╝ ██║   ██║██║     ██║   ██║   ██╔══╝  ██║     ██╔══██║██║╚██╗██║██║██╔═██╗ ██╔══██║    
██║     ╚██████╔╝███████╗██║   ██║   ███████╗╚██████╗██║  ██║██║ ╚████║██║██║  ██╗██║  ██║    
╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝   ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝"""
logo2 = """
██╗    ██╗██████╗  ██████╗  ██████╗ ██╗      █████╗ ██╗    ██╗███████╗██╗  ██╗ █████╗ 
██║    ██║██╔══██╗██╔═══██╗██╔════╝███║     ██╔══██╗██║    ██║██╔════╝██║ ██╔╝██╔══██╗
██║ █╗ ██║██████╔╝██║   ██║██║      ████║   ███████║██║ █╗ ██║███████╗█████╔╝ ███████║
██║███╗██║██╔══██╗██║   ██║██║      ██║     ██╔══██║██║███╗██║╚════██║██╔═██╗ ██╔══██║
╚███╔███╔╝██║  ██║╚██████╔╝╚██████╗ ███████╗██║  ██║╚███╔███╔╝███████║██║  ██╗██║  ██║
 ╚══╝╚══╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝"""


def loading(i=0): # recursive cool loading animation
    for x in range(i):
        print(".", end="", flush=True)
        time.sleep(0.25)
    print("\n" * 100)
    if i >= 3:
        return None
    else:pass
    i += 1
    loading(i)

if __name__ == "__main__": # what to do on import
    loading(1)
    time.sleep(2)
    print("Artificial Inteligence loaded succesfully sir")
    print(Fore.WHITE + logo1)
    print(Fore.RED + logo2)
    print(Fore.GREEN)
    time.sleep(4)

else:
    loading(1)
    time.sleep(2)
    print("Artificial Inteligence module loaded succesfully sir")
    print(Fore.WHITE + logo1)
    print(Fore.RED + logo2)
    print(Fore.GREEN)
    time.sleep(1)
    
    