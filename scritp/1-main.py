import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from scipy.stats import linregress, pearsonr, spearmanr
import logging
import os
import re
import joblib

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath, sep='\t')
    data = data.round(3)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    return X, y

def hyperparameter_search(X, y, log_directory, cv_strategy):
    models = {
        'KNN': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': range(1, 21),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=1),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [None, 10, 20, 30]
            }
        },
        'SVM': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'ElasticNet': {
            'model': ElasticNet(),
            'params': {
                'alpha': [0.1, 1, 10, 100],
                'l1_ratio': [0.1, 0.5, 0.7, 1.0]
            }
        },
        'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=1),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
    }

    if cv_strategy == 'loo':
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=1)

    best_estimators = {}
    all_results = []

    for name, spec in models.items():
        grid_search = GridSearchCV(spec['model'], spec['params'], cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X, y)
        best_estimators[name] = grid_search.best_estimator_

        for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
            model_instance = spec['model'].__class__(**params)
            model_instance.fit(X, y)  
            param_str = "_".join(f"{k}={v}" for k, v in params.items())
            filename = os.path.join(log_directory, f'{name}_{param_str}.joblib')
            joblib.dump(model_instance, filename)
            logging.info(f'{name} Params: {params}, Mean Score: {mean_score}, Model Saved as: {filename}')
            all_results.append({
                'model_name': name,
                'params': params,
                'mean_score': mean_score
            })

        logging.info(f'{name} best params: {grid_search.best_params_}, best score: {grid_search.best_score_}')

    return best_estimators, all_results


def calculate_correlations(y, y_pred):
    pearson_corr, pearson_p = pearsonr(y, y_pred)
    spearman_corr, spearman_p = spearmanr(y, y_pred)
    slope, intercept, r_value, p_value, std_err = linregress(y, y_pred)
    return {
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'linreg_r': r_value,
        'linreg_p': p_value,
        'linreg_slope': slope,
        'linreg_intercept': intercept,
        'linreg_stderr': std_err
    }

def plot_pearson(model, X, y, model_name, log_directory, phase, num_bacteria):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    pearson_corr, pearson_p_value = pearsonr(y, y_pred)
    
    spearman_corr, spearman_p_value = spearmanr(y, y_pred)
    
    slope, intercept, _, _, _ = linregress(y, y_pred)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    
    plt.plot(y, intercept + slope * y, 'green', label=f'Linear fit R² = {r2:.2f}\n' \
        f'Pearson R = {pearson_corr:.2f}, p-value = {pearson_p_value:.3g}\n' \
        f'Spearman R = {spearman_corr:.2f}, p-value = {spearman_p_value:.3g}')
    
    plt.xlabel('Real butyrate concentration (mmol/L)')  
    plt.ylabel('Predicted butyrate concentration (mmol/L)') 
    plt.title(f'Butyrate production correlation in simulated data ({num_bacteria} bacteria)')  
    plt.legend(loc='upper left')
    
    plot_directory = os.path.join(log_directory, 'plots')
    os.makedirs(plot_directory, exist_ok=True)
    
    plt.savefig(os.path.join(plot_directory, f'{phase}_{model_name}_pearson_actual_vs_predicted.png'))
    plt.close()

def save_results_to_new_file(results, log_directory, filename):
    results_file = os.path.join(log_directory, filename)
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False, encoding='utf-8')
    logging.info(f"Results saved to new file: {results_file}")

def main(cv_strategy='kfold'):
    file_names = [f for f in os.listdir() if f.endswith('bactS')]
    for file_name in file_names:
        num_bacteria = int(re.search(r'\d+', file_name).group())
        log_directory = f"{file_name[:-5]}_logs"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
        
        setup_logging(os.path.join(log_directory, 'setup_model_training.log'))
        
        filepath = file_name  
        X, y = load_and_prepare_data(filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

        setup_logging(os.path.join(log_directory, 'training_model_training.log'))
        _, train_results = hyperparameter_search(X_train, y_train, log_directory, cv_strategy)
        save_results_to_new_file(train_results, log_directory, 'training_results.csv')

if __name__ == '__main__':
    import sys
    cv_method = 'kfold'  
    if len(sys.argv) > 1:
        cv_method = sys.argv[1]
    main(cv_method)
