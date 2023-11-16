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

"""
# Get open_desk_4    ex. features and target
"""
place ='open_desk_4'
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
# Forcasting : Times series Prophet : 
"""


def prophet_forecast(df, target, other_features):
    # Rename columns for Prophet
    df = df.rename(columns = {'timestamp [dd/mm/yyyy HH:MM]' : 'ds', target: 'y'})
    df["ds"] = pd.to_datetime(df["ds"], format = "%d/%m/%Y %H:%M")
    print("check data processing : ")
    print("First timestamp in dataframe:", df['ds'].min())
    print("Last timestamp in dataframe:", df['ds'].max())
    full_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='H')
    missing_timestamps = full_range.difference(df['ds'])
    print("Missing timestamps:", missing_timestamps)
    
    # Plot target
    plt.figure(figsize=(12,6))
    plt.plot_date(df['ds'], df['y'], '-')
    plt.title('Target over time')
    plt.show()

    # Plot other features
    for feature in other_features:
        plt.figure(figsize=(12,6))
        plt.plot_date(df['ds'], df[feature], '-')
        plt.title(f'{feature} over time')
        plt.show()
    
    # Train & test split data 
    Date_split = '2013-10-01'
    train = df[df['ds'] <= Date_split] 
    test = df[df['ds'] > Date_split] 
    print(test.head(), test.columns)
    
    # Sort the data by date
    df = df.sort_values('ds')

    # Get the total number of data points
    n = len(df)

    # Compute the indices for the train/validation/test split
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.8)

    # Split the data
    df_train = df[:train_idx]
    df_val = df[train_idx:val_idx]
    df_test = df[val_idx:]
    
    print("df_train","df_val","df_test")
    print(df_train['ds'].min(),  df_val['ds'].min(), df_test['ds'].min())
    
    # Define initial and horizon
    initial = str((train['ds'].max() - train['ds'].min()).days) + ' days'
    horizon = str((test['ds'].max() - test['ds'].min()).days) + ' days'
    horizon_hours = str((test['ds'].max() - test['ds'].min()).days * 24) + ' hours'
    
    print("initial", "horizon" )
    print(initial, horizon )
    # Create Prophet model and add regressors
    m = Prophet(changepoint_prior_scale='0.01', seasonality_prior_scale='0.01', seasonality_mode='multiplicative', yearly_seasonality=True, weekly_seasonality=True, changepoint_range=0.9)
    m.add_country_holidays(country_name='AT')
    for feature in other_features:
        m.add_regressor(feature)

    # Fit model
    m.fit(train)

    # Create future dataframe and predict
    # future = m.make_future_dataframe(periods=30*24, freq='H')
    # # future_other_features = df[other_features]  # Make sure this includes future values
    # # future = pd.concat([future, future_other_features], axis=1)
    # forecast = m.predict(future)
    
    # Create future dataframe and predict
    # future = m.make_future_dataframe(periods=len(test), freq='H')
    # future = pd.DataFrame({
    # 'ds': future_timestamp_data,  # future dates for which you want to forecast temperature
    # 'humidity': future_humidity_data,  # estimated future humidity values
    # 'occupancy': future_occupancy_data,  # estimated future occupancy values
    # 'air_quality': future_air_quality_data  # estimated future air quality values
    # })
    
    future_other_features = test[['ds'] + other_features]
    future = test.drop('y', axis=1)
    print("future", future)
    

    # Predict and plot forecast
    forecast = m.predict(future)
    
    # forecast = m.predict(test)
    # print(forecast)
    
    # Plot forecast and components
    fig = m.plot(forecast)
    fig = m.plot_components(forecast)

    # Evaluate model
    cv = cross_validation(model=m, initial=initial, horizon=horizon, period='15 days')
    metrics = performance_metrics(cv)
    
    return metrics, forecast


"""_summary_
"""
humidity = model['features'][0]
other_features = model['features'][1:] + [model['target']] + global_features
target = humidity
metrics, forecast = prophet_forecast(df=merged_df, target=target, other_features=other_features)



### Only historical pattern of the temperature or Humidity : 
### Prophet : forcasting time series package: sesonality, Holidays, special Events


def forcasting(df, time_forcast_in_days, target ): 
    # Rename columns for Prophet
    df = df.rename(columns = {'timestamp [dd/mm/yyyy HH:MM]' : 'ds', target: 'y'})
    df["ds"] = pd.to_datetime(df["ds"], format = "%d/%m/%Y %H:%M")
    print("check data processing : ")
    print("First timestamp in dataframe:", df['ds'].min())
    print("Last timestamp in dataframe:", df['ds'].max())
    full_range = pd.date_range(start=df['ds'].min(), end=df['ds'].max(), freq='H')
    missing_timestamps = full_range.difference(df['ds'])
    print("Missing timestamps:", missing_timestamps)
    
    df = df[["ds","y"]]

    df.set_index('ds', inplace=True)
    df = df.resample('H').mean()  # or use .sum(), .max(), etc. depending on your needs
    df.reset_index(inplace=True)

    # Plot target
    plt.figure(figsize=(12,6))
    plt.plot_date(df['ds'], df['y'], '-')
    plt.title('Target over time')
    plt.show()
    
    # # Define initial and horizon
    # # Specify the length of the initial training period and the prediction horizon
    # horizon = str(24 * time_forcast_in_days )+ ' hours' 
    # initial = str(((len(df)) - (24 * time_forcast_in_days))/2) + ' hours'
    # horizon = str(24 * time_forcast_in_days )+ 'hours' 



    m = Prophet(changepoint_prior_scale='0.01',seasonality_prior_scale = '0.01',
                seasonality_mode='multiplicative',
                yearly_seasonality=True, weekly_seasonality=True,
                changepoint_range=0.9)
    # m = Prophet()
    m.add_country_holidays(country_name='AT') ### country_name='FR'for france
    m.fit(df)

    future = m.make_future_dataframe(periods=24 * time_forcast_in_days, freq='H')

    # Prediction & Vizualisation : 
    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    fig = m.plot(forecast)
    fig = m.plot_components(forecast)

    total_length = len(df)
    initial_length = int(total_length * 0.8)  # Use 80% of data for training
    horizon_length = total_length - initial_length  # Remaining 20% for forecasting

    initial = str(initial_length) + ' hours'
    horizon = str(horizon_length) + ' hours'
    # Evaluation : 
    from prophet.diagnostics import performance_metrics
    cv = cross_validation(model=m,initial=initial, horizon =horizon)
    metrics = performance_metrics(cv)
    print(metrics)
    from prophet.plot import plot_cross_validation_metric
    fig = plot_cross_validation_metric(cv, metric='mape') 


## Hyper-parameters tuning : # Python
def hyper_tuning(df): 
    import itertools
    import numpy as np
    import pandas as pd
    df = df.rename(columns = {'timestamp [dd/mm/yyyy HH:MM]' : 'ds', target: 'y'})
    df["ds"] = pd.to_datetime(df["ds"], format = "%d/%m/%Y %H:%M")
    total_length = len(df)
    initial_length = int(total_length * 0.8)  # Use 80% of data for training
    horizon_length = total_length - initial_length  # Remaining 20% for forecasting

    initial = str(initial_length) + ' hours'
    horizon = str(horizon_length) + ' hours'
    param_grid = {  
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m2 = Prophet(**params).fit(df)  # Fit model with given params
        df_cv = cross_validation(m2, initial=initial, horizon= horizon, parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    print(tuning_results)
    best_params = all_params[np.argmin(rmses)]
    print(best_params)
    
    
target = 'o4 [%]'
hyper_tuning(merged_df)
forcasting(merged_df,20, target)