

Overview

This project focuses on developing regression models to predict butyrate concentrations from microbiome datasets. It employs various machine learning algorithms, performs hyperparameter tuning, and evaluates model performance using statistical metrics and visualizations.


Scripts Description

Script 1: Model Training and Hyperparameter Tuning

Purpose:

- Load and preprocess microbiome data.
- Train multiple regression models with hyperparameter tuning using cross-validation.
- Save the trained models and log their performance metrics.

Key Functions:

1. setup_logging(log_file)

   - Configures logging to record details of the training process into a specified log file.

2. load_and_prepare_data(filepath)

   - Loads data from a CSV file and rounds numerical data to three decimal places.
   - Splits the data into features (X) and target variable (y).

3. hyperparameter_search(X, y, log_directory, cv_strategy)

   - Defines a set of regression models (KNN, RandomForest, SVM, ElasticNet, XGBoost) with respective hyperparameter grids.
   - Performs hyperparameter tuning using GridSearchCV with a specified cross-validation strategy (KFold or LeaveOneOut).
   - Saves each trained model with different hyperparameters.
   - Logs performance metrics and best parameters.
   - Returns the best estimators and a summary of all results.

4. calculate_correlations(y, y_pred)

   - Computes statistical correlations (Pearson, Spearman) and linear regression statistics between actual and predicted values.

5. plot_pearson(model, X, y, model_name, log_directory, phase, num_bacteria)

   - Generates scatter plots of actual vs. predicted values.
   - Includes linear fit line and correlation metrics on the plot.
   - Saves the plot in the specified directory.

6. save_results_to_new_file(results, log_directory, filename)

   - Saves the training results to a new CSV file.
   - Logs the file-saving operation.



Script 2: Results Visualization and Aggregation

Purpose:

- Aggregate testing results from different models and datasets.
- Filter and organize model parameters for analysis.
- Generate visualizations to compare model performance.

Key Steps:

1. Data Aggregation:

   - Scans all directories ending with '_logs' to find testing results.
   - Reads 'testing_results.csv' from each log directory and merges them into a single DataFrame.

2. Parameter Filtering:

   - Defines filter_params(params, model_name) to extract relevant hyperparameters for each model.
   - Applies this function to filter and clean the 'params' column.

3. Data Preparation:

   - Renames 'RandomForest' to 'RF' in the 'model_name' column for brevity.
   - Sorts the DataFrame based on the number of bacteria.
   - Saves the cleaned and merged data to 'merged_filtered_testing_results.csv'.

4. Visualization:

   - Defines create_heatmap() and create_combined_heatmap() to generate heatmaps of model performance metrics.
   - Metrics considered include 'pearson_corr', 'spearman_corr', and 'linreg_r'.
   - Generates individual heatmaps for each unique number of bacteria.
   - Creates a combined heatmap showcasing performance across all models and bacteria counts.
   - Saves the visualizations as PNG files.

---

Script 3: Model Testing and Performance Evaluation

Purpose:

- Load saved models and test them on new datasets.
- Calculate performance statistics.
- Summarize and save the evaluation results.

Key Steps:

1. Data Loading:

   - Defines load_test_data(filepath) to load test datasets.
   - Searches for test data files corresponding to specific numbers of bacteria

2. Model Prediction:

   - For each specified number of bacteria:
     - Loads the test data.
     - Initializes an empty DataFrame to store predictions.
     - Iterates over saved models in the corresponding directory (e.g., 'descartes/{consortia}/').
     - Loads each model and generates predictions on the test data.
     - Stores predictions along with the bacteria number in the DataFrame.
     - Saves the predictions to an Excel file ('{consortia}_descartes.xlsx').

3. Performance Evaluation:

   - Reads a manually prepared summary file 'summary_models_for_testing.csv'.
   - Cleans the data (e.g., replaces commas with dots, removes unnecessary columns).
   - Converts all data to numeric format.

4. Statistical Analysis:

   - Defines calculate_statistics_by_bacteria_number(df):
     - Groups the DataFrame by the number of bacteria.
     - For each group, calculates:
       - Pearson Correlation
       - Spearman Correlation
       - Kendall Tau
       - Mean Squared Error (MSE)
       - Median Squared Error
     - Stores the statistics in a nested dictionary.
   - Compiles the results into a DataFrame.
   - Saves the performance summary to 'models_performance_summary.xlsx'.


Usage Instructions

Ensure you have the following packages installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib
- scipy

Install the required packages using pip:

pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib scipy

