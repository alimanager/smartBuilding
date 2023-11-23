import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import holidays
from sklearn.cluster import KMeans
from tabulate import tabulate
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import datetime
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Specify your GitHub path
path = "https://raw.githubusercontent.com/alimanager/smartBuilding/master/"

# Get a list of all .csv files in the GitHub directory
# Please note that GitHub API doesn't allow listing directory contents, 
# so you have to provide the names of your csv files manually
csv_files = ['01_occ.csv', '02_win.csv', '03_light.csv','04_plug.csv','05_temp_in.csv','06_rhu_in.csv','07_rad_global.csv','08_temp_out.csv','09_rhu_out.csv','10_wsp.csv','11_wdi.csv']  # Replace with your actual file names

# Create a dictionary to store DataFrames with corresponding names for each .csv file
dfs = {}

# Read each .csv file, rename the DataFrame, and store it in the dictionary
for file_name in csv_files:
    df_name = file_name.replace('.csv', '')  # Extract DataFrame name from the file name
    url = path + file_name
    dfs[df_name] = pd.read_csv(url)  # Create DataFrame with the extracted name

    # Perform the merge based on a common key (e.g., 'common_column')
    # Replace 'common_column' with the actual column name that is common across all DataFrames
    merged_df = dfs['01_occ']  # Initialize merged_df with one of the DataFrames
    for df_name, df in dfs.items():
        if df_name != '01_occ':  # Skip the first DataFrame since it's already stored in merged_df
            merged_df = pd.merge(merged_df, df, on='timestamp [dd/mm/yyyy HH:MM]',how='outer' )
            # merged_df.fillna(method='ffill', inplace=True)
            # Forward-fill missing temperature values
            merged_df.fillna(method='ffill', inplace=True)


    # Now you have a merged DataFrame named 'merged_df' containing data from all .csv files
    nan_df = merged_df.isna()

    # If there are any NaN values, the nan_df DataFrame will contain True in those positions.
    # You can check if there are any NaN values in the entire DataFrame by using the any() method.
    if nan_df.any().any():
        print("The DataFrame contains NaN values.")
    else:
        print("The DataFrame does not contain NaN values.")


    merged_df

# variables selection, models to train/test : 
# Define global features
global_features = ['tempOut [C]', 'rh [%]', 'gh [W/m2]', 'wind speed [m/s]']

# Define models
models = {
    'kitchen': {
        'features': ['ki [%]', 'ki [C]', 'ki  [1:closed 0:open]', 'ki [0:off 1:on]'],
        'target': 'ki [0:vacant 1:occupied]'
    },
    'open_desk_4': {
        'features': ['o4 [%]', 'o4 [C]', 'o4_1 [1:closed 0:open]', 'o4_2 [1:closed 0:open]', 'o4_1 [0:off 1:on]', 'o4_2 [0:off 1:on]'] ,
        'target': 'o4 [0:vacant 1:occupied]'
    },
    # 'meeting_room': {
    #     'features': ["mr_1 [1:closed 0:open]", "mr_2 [1:closed 0:open]", "mr_3 [1:closed 0:open]", "mr_4 [1:closed 0:open]", "mr_5 [1:closed 0:open]", "mr_6 [1:closed 0:open]", "mr [C]", "mr [%]"] ,
    #     'target': 'mr [0:vacant 1:occupied]'
    # },
    'closed_desk_3': {
        'features': ["o3 [%]" , "o3 [C]", "o3_1 [1:closed 0:open]", "o3_2 [1:closed 0:open]", "o3_3 [1:closed 0:open]", "o3_4 [1:closed 0:open]", "o3_1 [0:off 1:on]", "o3_2 [0:off 1:on]", "o3 [W]"] ,
        'target': 'o3 [0:vacant 1:occupied]'
    }
}
# Define models labels in french : 
labels_fr = {
    'kitchen': {
        'features': ['Humidité Cuisine [%]', 'Température Cuisine [C]', 'Fenêtre Cuisine [1:fermée 0:ouverte]', 'Lumière Cuisine [0:éteinte 1:allumée]'],
        'target': 'Occupation Cuisine [0:vacant 1:occupé]'
    },
    'open_desk_4': {
        'features': ['Humidité Bureau Ouvert 4 [%]', 'Température Bureau Ouvert 4 [C]', 'Fenêtre Bureau Ouvert 4_1 [1:fermée 0:ouverte]', 'Fenêtre Bureau Ouvert 4_2 [1:fermée 0:ouverte]', 'Lumière Bureau Ouvert 4_1 [0:éteinte 1:allumée]', 'Lumière Bureau Ouvert 4_2 [0:éteinte 1:allumée]'] ,
        'target': 'Occupation Bureau Ouvert 4 [0:vacant 1:occupé]'
    },
    'closed_desk_3': {
        'features': ["Fenêtre Bureau Fermé 3_1 [1:fermée 0:ouverte]", "Fenêtre Bureau Fermé 3_2 [1:fermée 0:ouverte]", "Fenêtre Bureau Fermé 3_3 [1:fermée 0:ouverte]", "Fenêtre Bureau Fermé 3_4 [1:fermée 0:ouverte]", "Lumière Bureau Fermé 3_1 [0:éteinte 1:allumée]", "Lumière Bureau Fermé 3_2 [0:éteinte 1:allumée]", "Energie Bureau Fermé 3 [W]", "Température Bureau Fermé 3 [C]", "Humidité Bureau Fermé 3 [%]"] ,
        'target': 'Occupation Bureau Fermé 3 [0:vacant 1:occupé]'
    }
}


# !!! to be changed : open_desk_4 is selected !!!
place ='open_desk_4'

# !!! then the target and features will be selected accordinly :
model = models[place]
# Occupancy : 
model_features = model['features']
model_target = model['target']

# Humidity : 
Humidity = model['features'][0]
other_features_humd = model['features'][1:] + [model['target']]

# temperature : 
temperature = model['features'][1]
other_features_temp = model['features'][:1] + model['features'][2:] + [model['target']]

"""
#models french labels : 
"""
model_label = labels_fr[place]
model_target_label = model_label['target']

"""_summary_
    for A classification problem, convert Occupency to 0/1 
Returns:
    _type_: _description_
"""
# Categorize the target variable into classes : 
def categorize_presence(value):
    if value == 0.0:
        return 0.0
    else:
        return 1.0

# filter working hours : 
    """_summary_

    Returns:
        _type_: _description_
    """
import mlflow
 
def Classification(df,target,features,best_by="Accuracy", place=None):
    from pycaret.classification import setup,compare_models,pull, create_model, tune_model, predict_model, plot_model, evaluate_model, finalize_model, ClassificationExperiment

    Model = setup(df[features], target= df[target].apply(categorize_presence), fix_imbalance=True ,preprocess=False,  normalize=True, normalize_method='zscore')
    # Benchmarking the algorithmes for classification the target with pycaret : 
    # For Accuracy:
    if best_by == "Accuracy":
        # setup(df[features], target= df[target].apply(categorize_presence), fix_imbalance=True ,preprocess=False,  normalize=True, normalize_method='zscore', log_experiment = True, experiment_name =place )
        best_model = compare_models(sort='Accuracy')
    # For Recall:
    elif best_by == "Recall":
        best_model = compare_models(sort='Recall')
    else:
        print(f"Invalid value for best_by: {best_by}. Defaulting to 'Accuracy'.")
        best_model = compare_models(sort='Accuracy')
            

    print(best_model)
    results_model2 = pull()

    # if intersted in other model by: execution time, interpretability or by other metric 
    # for example if we choose the random forest classifier : rf_classifier
    # rf_classifier = create_model('rf') # can added parameters of the model before tuning
    # model_tuned = tune_model(rf_classifier)
    # else : the best_model automaticly chossen 
    model_tuned = tune_model(best_model)

    # Prediction model : 
    predict_model2= predict_model(model_tuned);
    print(predict_model2)
    # plotting Prediction : 
    ## Evaluation model 2: 

    plot_model(model_tuned, plot = 'auc')
    plot_model(model_tuned, plot = 'feature')
    plot_model(model_tuned, plot = 'confusion_matrix')

    # evaluate_model(model_tuned)
    results_model2 = pull()
    print(results_model2)

    ## Production store model : 
    #Final Random Forest model parameters for deployment
    final_model = finalize_model(model_tuned);
    print(final_model)
    
    #for Production new data : 
    # predictions = predict_model(final_model, data=data_unseen)
    # print(predictions.head())
    
    return final_model 

def regression(df,target,features,best_by="R2", place=None):
    from pycaret.regression import setup,compare_models,pull, create_model, tune_model, predict_model, plot_model, evaluate_model, finalize_model

    Model = setup(df[features], target= df[target], preprocess=False,  normalize=True, normalize_method='zscore')
    # Benchmarking the algorithmes for regression the target with pycaret : 
    if best_by == "R2":
        best_model = compare_models(sort='R2')  # sorts by R2
    elif best_by == "RMSE":
        best_model = compare_models(sort='RMSE')  # sorts by RMSE
    else:
        print(f"Invalid value for best_by: {best_by}. Defaulting to 'R2'.")
    best_model = compare_models(sort='R2')

  
    print(best_model)
    results_model2 = pull()

    # if intersted in other model by: execution time, interpretability or by other metric 
    # for example if we choose the random forest classifier : rf_classifier
    # rf_classifier = create_model('rf') # can added parameters of the model before tuning
    # model_tuned = tune_model(rf_classifier)
    # else : the best_model automaticly chossen 
    model_tuned = tune_model(best_model)

    # Prediction model : 
    predict_model2= predict_model(model_tuned);
    print(predict_model2)
    # plotting Prediction : 
    ## Evaluation model 2: 

    # Residual Plot
    plot_model(model_tuned, plot = 'residuals')
    # Prediction Error Plot
    plot_model(model_tuned, plot = 'error')
    # Feature Importance Plot
    plot_model(model_tuned, plot = 'feature')
    plot_model(model_tuned, plot = 'learning')
    plot_model(model_tuned, plot = 'vc')
    plot_model(model_tuned, plot = 'cooks')

    # evaluate_model(model_tuned)
    results_model2 = pull()
    print(results_model2)

    ## Production store model : 
    #Final Random Forest model parameters for deployment
    final_model = finalize_model(model_tuned);
    print(final_model)
    
    #for Production new data : 
    # predictions = predict_model(final_model, data=data_unseen)
    # print(predictions.head())
    
    return final_model 

"""
# Get open_desk_4    ex. features and target
# !! important !! : change place to be modelised from the dictionnary above !!!: 
# !!! the model select in place is "open desk" : change it to
        - kitchen
        - open_desk_4 # selected
        - meeting_room  # is commented we don't have occupancy data
        - closed_desk_3
"""

Classification(merged_df,model_target, model_features,best_by="Recall", place=place)

# for kitchen occupancy : 
place ='kitchen'
Classification(merged_df,model_target, model_features, best_by="Recall", place=place)

# for closed_desk_3 occupancy : 
Classification(merged_df,model_target, model_features, best_by="Recall", place=place)

# Temperature : 
regression(merged_df,temperature, other_features_temp, best_by="RMSE", place=place)

# Humidity : 
regression(merged_df,Humidity, other_features_humd, best_by="RMSE", place=place)