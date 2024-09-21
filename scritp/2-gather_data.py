import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast

log_directories = [d for d in os.listdir() if d.endswith('_logs')]
merged_df = pd.DataFrame()

for log_dir in log_directories:
    file_path = os.path.join(log_dir, 'testing_results.csv')
    if os.path.exists(file_path):
        df_temp = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df_temp], ignore_index=True)
    else:
        print(f"Alert: File {file_path} not found and will be skipped.")

def filter_params(params, model_name):
    params = params.replace('None', 'None').replace('nan', 'None')  
    try:
        params_dict = ast.literal_eval(params)
    except ValueError as e:
        print(f"Error parsing params: {params}")
        return {}

    model_params = {
        'KNN': ['metric', 'n_neighbors', 'weights'],
        'RandomForest': ['max_depth', 'max_features', 'n_estimators'],
        'SVM': ['C', 'gamma', 'kernel'],
        'ElasticNet': ['alpha', 'l1_ratio'],
        'XGBoost': ['n_estimators', 'learning_rate', 'max_depth'],
        'LinearRegression': ['fit_intercept', 'positive']
    }

    return {k: v for k, v in params_dict.items() if k in model_params.get(model_name, [])}

merged_df['filtered_params'] = merged_df.apply(lambda row: filter_params(row['params'], row['model_name']), axis=1)
merged_df['filtered_params'] = merged_df['filtered_params'].apply(lambda x: str(x))

merged_df['model_name'] = merged_df['model_name'].replace('RandomForest', 'RF')

merged_df = merged_df.sort_values(by='num_bacteria')
output_file_filtered = 'merged_filtered_testing_results.csv'
merged_df.to_csv(output_file_filtered, index=False)

all = pd.read_csv("merged_filtered_testing_results.csv")
metrics = ['pearson_corr', 'spearman_corr', 'linreg_r']
def create_heatmap(df, metrics, title, filename):
    plt.figure(figsize=(5, 5))
    sns.heatmap(df.pivot_table(index="model_name", columns="phase", values=metrics).T, annot=True, cmap="YlGnBu", cbar_kws={'shrink': 0.6})
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
unique_bacteria_counts = all['num_bacteria'].unique()
for count in unique_bacteria_counts:
    df_filtered = all[all['num_bacteria'] == count]
    create_heatmap(df_filtered, metrics, f"Model Performance for {count} bacteria", f"model_performance_{count}_bacteria.png")

def create_combined_heatmap(df, metrics, title, filename):
    data = df[metrics]
    plt.figure(figsize=(3, 12))
    sns.heatmap(data, annot=True, cmap="YlGnBu", xticklabels=metrics, yticklabels=df['model_name'] + " (" + df['phase'] + "," + df['num_bacteria'].astype(str) + ")", cbar_kws={'shrink': 0.6})
    plt.title(title)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

create_combined_heatmap(all, metrics, "Models Performance", "models_performance.png")
