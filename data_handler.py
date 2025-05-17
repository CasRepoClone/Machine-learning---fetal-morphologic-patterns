#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# please start venv before running this script and installing the required packages
#
# .\.venv\Scripts\activate
#
"""      This module handles the data processing and cleaning            """
import pandas as pd 

def print_frame(df):
    print(df.head())

def clean_data(df) -> pd.DataFrame:
    """Clean the data by forward filling, then filling remaining NaNs with column mean"""
    df = df.fillna(method='ffill')
    df = df.fillna(df.mean(numeric_only=True))
    if df.isnull().values.any():
        print("Warning: There are still missing values in the dataset after cleaning.")
    return df
        
def load_training_Data(csv, remove) -> pd.DataFrame: 
    """Load the training data from a CSV file. remove the last 'remove' number of rows."""
    df = pd.read_csv(csv)
    return df.iloc[:-remove]

def load_validation_Data(csv, trainingDataSize) -> pd.DataFrame:
    """Load the validation data from a CSV file. keep the last 'trainingDataSize' number of rows."""
    df = pd.read_csv(csv)
    return df.iloc[-trainingDataSize:]
#endregion