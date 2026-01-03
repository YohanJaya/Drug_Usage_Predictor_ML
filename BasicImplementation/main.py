"""
Main script to run the complete drug usage prediction pipeline
Linear Regression with Gradient Descent
Now using combined real (Hyper.csv) + synthetic data
"""

import numpy as np
from src.data import  split_data, read_data
from src.models import train_model, predict, evaluate_model
from src.utils import plot_cost_history, plot_predictions, plot_residuals, plot_feature_importance

def main():

    # 1. Read data
    print("1. Reading data...")
    filePath = 'data/hospital_drug_demand.csv'
    df = read_data(filePath)
   
   # 2. Split data
    print("2. Splitting data...")
    xTrain, yTrain, xTest, yTest = split_data(df)
    
    # 3. Train model
    print("3. Training model...")
    model = train_model(xTrain, yTrain)
    
    
    # 4. predictions
    print("4. Making predictions...")
    predictions = predict(model, xTest)
    
    # 5. Evaluate model
    print("5. Evaluating model...")
    evaluation_metrics = evaluate_model(yTest, predictions)
    
    

    
    # 6. Visualize results
    print("6. Visualizing results...")
    plot_cost_history(model['cost_history'])
    plot_predictions(yTest, predictions)
    plot_residuals(yTest, predictions)
    plot_feature_importance(model, df.columns[:-1])
    

    