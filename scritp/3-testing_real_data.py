import numpy as np
import joblib
import re
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error


def load_test_data(filepath):
    data = pd.read_csv(filepath, sep='\t')
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    return X, y


numbers = [2,3,13]

for consortia in numbers:
    for txt in os.listdir():
        if txt.endswith(f"_{consortia}.txt"):
            X_test, y_test = load_test_data(txt)
    predictions_df = pd.DataFrame()

    for model_file in os.listdir(f"descartes/{consortia}/"):
        model_name = model_file.split('_full_data_model_')[0]
        model = joblib.load(F"descartes/{consortia}/{model_file}")
        predictions_df["bacteria number"] = consortia
        y_pred = model.predict(X_test)
        predictions_df[model_name] = y_pred

    predictions_df.to_excel(f"{consortia}_descartes.xlsx")

#manual extraction of summary for each model --> summary_models_for_testing.csv

descartes = pd.read_csv('summary_models_for_testing.csv',sep=";")
df = descartes.replace({',': '.'}, regex=True)
del df['Samples']
del df['Unnamed: 163']
df = df.apply(pd.to_numeric, errors='coerce')


def calculate_statistics_by_bacteria_number(df):
    bacteria_groups = df.groupby('bacteria number')
    results = {}
    
    for group_name, group_df in bacteria_groups:
        group_results = {}
        columns = df.columns.tolist()[1:]
        
        for col in columns:
            pearson_corr,  = pearsonr(group_df[col], group_df['real  [mmol/L]'])
            spearman_corr, _ = spearmanr(group_df[col], group_df['real  [mmol/L]'])
            kendall_corr, _ = kendalltau(group_df[col], group_df['real  [mmol/L]'])
            mse = mean_squared_error(group_df[col], group_df['real  [mmol/L]'])
            median_squared_error = mean_squared_error(group_df[col], group_df['real  [mmol/L]'], squared=False)  
            
            group_results[col] = {
                'Pearson Correlation': pearson_corr,
                'Spearman Correlation': spearman_corr,
                'Kendall Tau': kendall_corr,
                'Mean Squared Error': mse,
                'Median Squared Error': median_squared_error
            }
        
        results[group_name] = group_results
    
    return results

statistics_by_bacteria = calculate_statistics_by_bacteria_number(df)

results_df = pd.concat({k: pd.DataFrame(v).T for k, v in statistics_by_bacteria.items()}, axis=0)
results_df.to_excel('models_performance_summary.xlsx')
