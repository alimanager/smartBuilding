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
        return 0
    else:
        return 1

# filter working hours : 
    """_summary_

    Returns:
        _type_: _description_
    """
import mlflow

"""best_y : "Prec." , "Recall", "F1"
"""

def Classification(df,target,features,best_by, place=None,new_data=None):
    from pycaret.classification import setup,compare_models,pull, create_model, tune_model, predict_model, plot_model, evaluate_model, finalize_model, ClassificationExperiment

    Model = setup(df[features], target= df[target].apply(categorize_presence),preprocess=False,  normalize=True, normalize_method='zscore')
    # Benchmarking the algorithmes for classification the target with pycaret : 
    # For Accuracy:
    if best_by == "Accuracy":
        # setup(df[features], target= df[target].apply(categorize_presence), fix_imbalance=True ,preprocess=False,  normalize=True, normalize_method='zscore', log_experiment = True, experiment_name =place )
        best_model = compare_models(sort='Accuracy')
    # For Recall:
    elif best_by == best_by:
        best_model = compare_models(sort=best_by)
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
    if best_by != "Prec.":
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
    predictions = None
    if new_data  is not None and not new_data.empty:
        predictions = predict_model(final_model, data=new_data)
        print(predictions.columns)
        
        # # Visualize actual vs predicted values
        # visualize_predictions(predictions[target], predictions['prediction_label'],
        #                        f'{target} Prediction - {place}', 'Timestamp', target)
    predictions
    return final_model, predictions

    """_summary_
    Best_by = "R2", "RMSE"
    """
def regression(df,target,features,best_by="R2", place=None,new_data=None):
    from pycaret.regression import setup,compare_models,pull, create_model, tune_model, predict_model, plot_model, evaluate_model, finalize_model
    Model = setup(df[features], target= df[target], preprocess=False, normalize=True, normalize_method='zscore')
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
   

    # evaluate_model(model_tuned)
    results_model2 = pull()
    print(results_model2)

    ## Production store model : 
    #Final Random Forest model parameters for deployment
    final_model = finalize_model(model_tuned);
    print(final_model)
    
    #for Production new data : 
    predictions = None
    if new_data  is not None and not new_data.empty:
        predictions = predict_model(final_model, data=new_data)
        print(predictions.columns)
    
    return final_model , predictions

"""
# Get open_desk_4    ex. features and target
# !! important !! : change place to be modelised from the dictionnary above !!!: 
# !!! the model select in place is "open desk" : change it to
        - kitchen
        - open_desk_4 # selected
        - meeting_room  # is commented we don't have occupancy data
        - closed_desk_3
"""

# Assuming 'timestamp [dd/mm/yyyy HH:MM]' is in datetime format
merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp [dd/mm/yyyy HH:MM]'], format='%d/%m/%Y %H:%M')
merged_df.set_index('timestamp', inplace=True)

# Resample the data by hour
merged_df = merged_df.resample('H').mean()

# Filter data for December
december_data_hourly = merged_df[merged_df.index.month == 12]

# Training set (excluding December)
train_data_hourly = merged_df[merged_df.index.month != 12].copy()

# Test set (December only)
test_data_hourly = december_data_hourly.copy()

#Filter out ; working hours:::
# def filter_working_hours(df, week_end_too=None, holidays_too=None):
#     # df['timestamp [dd/mm/yyyy HH:MM]'] = pd.to_datetime(df['timestamp [dd/mm/yyyy HH:MM]'])
#     # df.set_index('timestamp [dd/mm/yyyy HH:MM]', inplace=True)
    
#     # Filter the data to include only the hours from 6 to 17
#     df_working_hours = df.between_time('6:00', '17:00')
    
#     if week_end_too == True:
#         # Identify weekends and public holidays in Austria
#         at_holidays = holidays.Austria(years=df_working_hours.index.year.unique())
#         df_working_hours['is_weekend_or_holiday'] = df_working_hours.index.to_series().apply(lambda x: x.weekday() >= 5 or x in at_holidays)

#         # Filter out weekends and holidays from the analysis
#         df_working_hours = df_working_hours[~df_working_hours['is_weekend_or_holiday']]
#     # df_working_hours.drop("is_weekend_or_holiday")    
#     return df_working_hours

# test_data_hourly = filter_working_hours(test_data_hourly, week_end_too=True)
# train_data_hourly = filter_working_hours(train_data_hourly, week_end_too=True)

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from Data_loader import *

# Train and evaluate classification model
classification_model, classification_predictions = Classification(train_data_hourly, model_target, model_features+global_features, best_by="Accuracy", place=place,new_data=test_data_hourly)
classification_model, classification_predictions = regression(train_data_hourly,model_target, model_features + global_features, best_by="RMSE", place=place,new_data=test_data_hourly)

#save model : 
# save pipeline
from pycaret.classification import save_model, load_model,save_experiment
save_experiment('class_1_0_pipeline_1H_desk04')
save_model(classification_model, 'class_1_0_pipeline')
load_model('class_1_0_pipeline')


classification_predictions["prediction_label"].plot()
classification_predictions[model_target].apply(categorize_presence).plot()

classification_predictions["prediction_score"].plot()
classification_predictions[model_target].plot()

from sklearn.metrics import classification_report, confusion_matrix


y_pred = classification_predictions["prediction_label"]
y_true = classification_predictions[model_target].apply(categorize_presence)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))


# Create a figure with two traces
fig = go.Figure()

# Plot prediction_label with full opacity
fig.add_trace(go.Bar(x=classification_predictions['timestamp [dd/mm/yyyy HH:MM]'],
                     y=classification_predictions['prediction_label'],
                     name='Prediction Label', marker_color='blue'))

# Plot model_target with reduced opacity
fig.add_trace(go.Bar(x=classification_predictions['timestamp [dd/mm/yyyy HH:MM]'],
                     y=classification_predictions[model_target].apply(categorize_presence),
                    #  y=classification_predictions[model_target],
                     name='Model Target', marker_color='rgba(255, 0, 0, 0.5)'))

# Update layout
fig.update_layout(barmode='overlay',  # Overlay bars on the same axis
                  title='Bar Plot for Prediction Label and Model Target',
                  xaxis_title='Date',
                  yaxis_title='Values')

# Customize the x-axis to show dates nicely
fig.update_xaxes(
    tickformat='%d/%m/%Y %H:%M',
    tickangle=45,
    tickmode='auto'
)

# Show the plot
fig.show()


# for kitchen occupancy : 
place ='kitchen'
Classification(merged_df,model_target, model_features, best_by="Prec.", place=place)

# for closed_desk_3 occupancy : 
Classification(merged_df,model_target, model_features, best_by="Prec.", place=place)

# Temperature : 
# regression(merged_df,temperature, other_features_temp + global_features, best_by="RMSE", place=place)

regression_predictions_model_Temp, regression_predictions_Temp = regression(train_data_hourly,temperature, other_features_temp + global_features, best_by="RMSE", place=place ,new_data= test_data_hourly)

# save pipeline
from pycaret.regression import save_model, load_model,save_experiment,check_drift

save_experiment('regression_predictions_model_Temp')
save_model(regression_predictions_model_Temp, 'Temp_pipeline')
load_model('Temp_pipeline')

#plotting predict vs target
regression_predictions_Temp["prediction_label"].plot()
regression_predictions_Temp[temperature].plot()





# Humidity : 
# regression(merged_df,Humidity, other_features_humd + global_features, best_by="RMSE", place=place)

regression_predictions_model_humd, regression_predictions_humd = regression(train_data_hourly,Humidity, other_features_humd + global_features, best_by="RMSE", place=place ,new_data= test_data_hourly)

# save pipeline model :
from pycaret.regression import save_model, load_model,save_experiment
save_model(regression_predictions_model_humd, 'humd_pipeline')
load_model('humd_pipeline')
save_experiment(regression_predictions_model_humd)

#plotting predict vs target:
regression_predictions_humd["prediction_label"].plot()
regression_predictions_humd[Humidity].plot()


"""
# Plotting :: prédictions vs valeurs
"""

# Plotting
plt.figure(figsize=(12, 6))

# Plot true values
plt.plot(classification_predictions[model_target].apply(categorize_presence), label='True Values', marker='o')

# Plot predictions
plt.plot(classification_predictions["prediction_label"], label='Predictions', marker='o')

plt.title('True Values vs Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()

"""# heat binary classification : """

import plotly.express as px
import pandas as pd

# Assuming you have a DataFrame named 'classification_predictions'
# Replace 'prediction_label' and 'model_target' with your actual column names

# Convert the 'timestamp' index to datetime with the correct format
classification_predictions.index = pd.to_datetime(classification_predictions.index, format='%d/%m/%Y %H:%M')

# Filter data for the month of December
december_data = classification_predictions[
    (classification_predictions.index.month == 12)
]

# Create a binary column indicating whether prediction matches label
december_data['match'] = (december_data['prediction_label'] == december_data[model_target].apply(categorize_presence)).astype(int)

# Get the range of days in December
days_in_december = range(1, 32)

# Pivot the data to get it in the form suitable for a heatmap, including all days
heatmap_data = december_data.pivot_table(index=december_data.index.day,
                                         columns=december_data.index.hour*4 +
                                         december_data.index.minute/15,
                                         values='match',
                                         aggfunc='max',
                                         fill_value=0,
                                         dropna=False  # Include all days even if some are missing
                                         ).reindex(index=days_in_december, fill_value=0)

# Create the heatmap with just two colors
fig = px.imshow(heatmap_data,
                labels=dict(color="Match"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                color_continuous_scale=["red", "green"],
                color_continuous_midpoint=0.5,  # Set midpoint to ensure two colors only
                title='Prediction occupiation in open-desk-4 Match Heatmap for December',
                )

fig.update_xaxes(
    title_text='Time (15-minute intervals)',
    tickvals=list(range(0, 96, 4)),
    ticktext=[f'{i // 4:02d}:{(i % 4) * 15:02d}' for i in range(0, 96, 4)],
    tickangle=45,
)

fig.update_yaxes(
    title_text='Day of December',
)

fig.show()



"""#plot Temp : """
import plotly.express as px
import pandas as pd

# Assuming 'merged_df' is your DataFrame
# merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp [dd/mm/yyyy HH:MM]'], format='%d/%m/%Y %H:%M')
# merged_df.set_index('timestamp', inplace=True)
rooms_Temp = merged_df.filter(like="C").columns


# Select the relevant columns for the heatmap
heatmap_df = merged_df[rooms_Temp]

# Transpose the DataFrame to have rooms on the y-axis and timestamps on the x-axis
heatmap_df = heatmap_df.T

# Create the heatmap
fig = px.imshow(
    heatmap_df, 
    labels=dict(x='Timestamp', y='rooms_Temp', color='Temperature'), 
    color_continuous_scale='RdBu_r', 
    origin='lower'
)

# Customize the layout if needed
fig.update_layout(
    title='Room Temperature Heatmap',
)

# Show the plot
fig.show()


#


import plotly.express as px
import pandas as pd

# Assuming 'merged_df' is your DataFrame
# merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp [dd/mm/yyyy HH:MM]'], format='%d/%m/%Y %H:%M')
# merged_df.set_index('timestamp', inplace=True)
rooms_Humd = merged_df.filter(like="%").columns


# Select the relevant columns for the heatmap
heatmap_df = merged_df[rooms_Humd]

# Transpose the DataFrame to have rooms on the y-axis and timestamps on the x-axis
heatmap_df = heatmap_df.T

# Create the heatmap
fig = px.imshow(
    heatmap_df, 
    labels=dict(x='Timestamp', y='rooms_Humd', color='Humidity'), 
    color_continuous_scale='RdBu_r', 
    origin='lower'
)

# Customize the layout if needed
fig.update_layout(
    title='Room Humidity Heatmap',
)

# Show the plot
fig.show()
