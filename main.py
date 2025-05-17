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

import csv # native csv module
import seaborn as sns
import pandas as pd # data manipulation and analysis
# user modules 
import data_handler as handler # we have created a new module called data_handler.py to handle the data processing and cleaning 
import ArtificialInteligence as ai
import matplotlib.pyplot as plt

# -- settings -- 
csv = 'cardiotocography_v2.csv'
trainingDataSize = 500

# load the training and validation data 
training = handler.load_validation_Data(csv, trainingDataSize)
validation = handler.load_training_Data(csv, remove=trainingDataSize)

# clean the data 
training = handler.clean_data(training)
validation = handler.clean_data(validation)

# running the models

hyperparameters = {'max_depth': 5, 'min_samples_split': 10, 'criterion': 'entropy'}
model, tree, normalizer, standadizer = ai.run(training, validation, hyperparameters)


def test_new_data(new_row): 
    """testing new data against the model (AI)"""
    feature_columns = training.drop(columns=['CLASS']).columns
    new_df = pd.DataFrame([new_row], columns=feature_columns)

    # Clean the new row 
    new_df = handler.clean_data(new_df)

    # Transform and predict (using normalizer as example)
    new_row_transformed = normalizer.transform(new_df)
    prediction = model.predict(new_row_transformed)
    proba = model.predict_proba(new_row_transformed)[0]

    print("CLASS - [target] fetal morphologic pattern class code (1 to 10)")
    print("Predicted CLASS:", prediction[0])

    # Heatmap visualization
    plt.figure(figsize=(8, 2))
    sns.heatmap([proba], annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=model.classes_, yticklabels=["Probability"])
    plt.xlabel("CLASS")
    plt.title(f"Predicted Class Probabilities for New data with a depth of: {hyperparameters['max_depth']}")
   
    plt.show()

# testing row for new data against the model (AI)
row = [
    130.0, 0.005, 0.0, 
    0.007, 0.002, 0.0, 
    0.0, 20.0, 2.0, 
    0.0, 12.0, 120.0, 
    60.0, 180.0, 4.0, 
    0.0, 130.0, 135.0, 
    132.0, 15.0, 0.0
]
test_new_data(row)