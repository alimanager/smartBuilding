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

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential

def Lstm_autoencoders(df,target, features, sequence):
    df_temp = df.copy()
    # Perform feature engineering on the df_temp DataFrame
    df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp [dd/mm/yyyy HH:MM]'], format='%d/%m/%Y %H:%M')
    df_temp.set_index('timestamp', inplace=True)
    df_temp = df_temp.resample('H').mean(numeric_only=True)

    # Create time-based features
    df_temp['hour'] = df_temp.index.hour
    df_temp['dayofweek'] = df_temp.index.dayofweek
    df_temp['quarter'] = df_temp.index.quarter
    df_temp['month'] = df_temp.index.month
    df_temp['year'] = df_temp.index.year
    df_temp['dayofyear'] = df_temp.index.dayofyear
    df_temp['dayofmonth'] = df_temp.index.day
    df_temp['weekofyear'] = df_temp.index.isocalendar().week
    df_temp.reset_index(inplace=True)
    
     # 2. Feature Engineering : 
    def Feature_Engineering(df,var):
        df[var+'_lag_1'] = df[var].shift(1)                          # Lag Features:the previous timestep's temperature for each data point.
        df[var+'_rolling_mean_3'] = df[var].rolling(window=3).mean() # Rolling mean over a window of 3 time periods
        df[var+'expanding_mean'] = df[var].expanding().mean()        # Expanding Window Features:  mean include all prior data points.
        df[var+'diff'] = df[var].diff()                              # Differencing : 
        df.fillna(method='ffill', inplace=True)
        
    # Apply the Feature_Engineering function to df_temp
    Feature_Engineering(df_temp, target)
    for feature in features:
        Feature_Engineering(df_temp, feature)
    
    # Split the data into training and testing sets
    train_size = int(len(df_temp) * 0.8)
    # train, test = df[:train_size], df[train_size:]
    train, test = df_temp[:train_size].copy(), df_temp[train_size:].copy()

    
    # Fit the scaler on the training set and transform both sets
    # train[target] = scaler.fit_transform(train[target].values.reshape(-1, 1))
    # test[target] = scaler.transform(test[target].values.reshape(-1, 1))

    # for feature in features:
    #     train[feature] = scaler.fit_transform(train[feature].values.reshape(-1, 1))
    #     test[feature] = scaler.transform(test[feature].values.reshape(-1, 1))

    # scaler = MinMaxScaler()
    # train[target] = scaler.fit_transform(train[target].values.reshape(-1, 1))
    # test[target] = scaler.transform(test[target].values.reshape(-1, 1))

    # for feature in features:
    #     train[feature] = scaler.fit_transform(train[feature].values.reshape(-1, 1))
    #     test[feature] = scaler.transform(test[feature].values.reshape(-1, 1))
    
    scaler = MinMaxScaler()
    train.loc[:, target] = scaler.fit_transform(train[target].values.reshape(-1, 1))
    test.loc[:, target] = scaler.transform(test[target].values.reshape(-1, 1))

    for feature in features:
        train.loc[:, feature] = scaler.fit_transform(train[feature].values.reshape(-1, 1))
        test.loc[:, feature] = scaler.transform(test[feature].values.reshape(-1, 1))
        

    # 1. Data Processing : 
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # df[target] = scaler.fit_transform( df[target].values.reshape(-1, 1))
    # for feature in features:
    #     df[feature] = scaler.fit_transform(df[feature].values.reshape(-1, 1))
    
   
        
    # Feature_Engineering(df_temp,target)
        
    # # for feature in features:
    # #     Feature_Engineering(df_temp,feature)
    
    # # Perform feature engineering independently on both sets
    # for feature in features:
    #     Feature_Engineering(train, feature)
    #     Feature_Engineering(test, feature)
        
    
    
    # 3. Reshape data for LSTM:
    
    # Reshape data : 
    def create_sequences(data, sequence_length):
        sequences = []
        for i in range(len(data) - sequence_length):
            seq = data.iloc[i:i+sequence_length]
            sequences.append(seq.values)
        return np.array(sequences) 

    sequence_length= sequence
    
    # 4. Train-Test Split: : 
    
#    # Split the data into training and testing sets
#     train_size = int(len(df) * 0.8)
#     train, test = df[:train_size], df[train_size:]
    
    # Create sequences
    # X_train = create_sequences(train.drop(['timestamp',target], axis=1), sequence_length)
    # y_train = create_sequences(train[[target]], sequence_length)
    # X_test = create_sequences(test.drop(['timestamp',target], axis=1), sequence_length)
    # y_test = create_sequences(test[[target]], sequence_length)

    # X_train = X_train.astype(float)
    # y_train = y_train.astype(float) 
    # X_test = X_test.astype(float)
    # y_test = y_test.astype(float)
    
    
    X_train = create_sequences(train[features], sequence_length)
    y_train = create_sequences(train[[target]], sequence_length)
    X_test = create_sequences(test[features], sequence_length)
    y_test = create_sequences(test[[target]], sequence_length)

    X_train = X_train.astype(float)
    y_train = y_train.astype(float) 
    X_test = X_test.astype(float)
    y_test = y_test.astype(float)


    print('is Na in train :',np.isnan(X_train).any())
    print('X_train shape :',X_train.shape)
    print('is Na in target :',np.isnan(y_train).any())
    print('Y_train shape :',y_train.shape)
    
    # # Create the LSTM autoencoder model
    # model = Sequential([
    #     LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    #     LSTM(5, activation='relu', return_sequences=False),
    #     RepeatVector(X_train.shape[1]),
    #     LSTM(5, activation='relu', return_sequences=True),
    #     LSTM(10, activation='relu', return_sequences=True),
    #     TimeDistributed(Dense(X_train.shape[2]))
    # ])

    # model.compile(optimizer='adam', loss='mse')
    # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    return  X_train,X_test,y_train,y_test, train ,test, scaler



def Lstm_autoencoder_model(X_train,y_train ,epochs=10,batch_size=32): 
     # Create the LSTM autoencoder model
    model = Sequential([
        LSTM(10, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
        LSTM(5, activation='relu', return_sequences=False),
        RepeatVector(X_train.shape[1]),
        LSTM(5, activation='relu', return_sequences=True),
        LSTM(10, activation='relu', return_sequences=True),
        TimeDistributed(Dense(X_train.shape[2]))
    ])

    model.compile(optimizer='adam', loss='mse',)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history,model

X_train,X_test,y_train,y_test,train ,test,scaler = Lstm_autoencoders(merged_df,temperature, other_features_temp,10)
history, model = Lstm_autoencoder_model(X_train,y_train, epochs=10, batch_size=32)


def Lstm_autoencoder_model(y_train, epochs=10, batch_size=32): 
    # Create the LSTM autoencoder model
    model = Sequential([
        LSTM(10, activation='relu', input_shape=(y_train.shape[1], y_train.shape[2]), return_sequences=True),
        LSTM(5, activation='relu', return_sequences=False),
        RepeatVector(y_train.shape[1]),
        LSTM(5, activation='relu', return_sequences=True),
        LSTM(10, activation='relu', return_sequences=True),
        TimeDistributed(Dense(y_train.shape[2]))
    ])

    model.compile(optimizer='adam', loss='mse',)
    history = model.fit(y_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)
    return history, model

history, model = Lstm_autoencoder_model(y_train, epochs=10, batch_size=32)



def evaluate_model(history):
    # Plot the training and validation loss
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()

evaluate_model(history)


# # Get the optimal hyperparameters
# best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

# # Build the model with the optimal hyperparameters
# model = tuner.hypermodel.build(best_hps)

# # Train the model on the full training data
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

    
def evaluate_and_plot(model, X_test, y_test, y_train, sequence_length, plot_metric='mae',pourcentage=90):
    # Make predictions (reconstruct the input) on the test data
    y_test_pred = model.predict(y_test)

    # Calculate the MSE and MAE between the original and reconstructed data
    mse_y_test = np.mean(np.power(y_test_pred - y_test, 2), axis=1)
    mae_y_test = np.mean(np.abs(y_test_pred - y_test ), axis=1)

    # Print the mean MSE and MAE
    print(f'Mean MSE Test: {np.mean(mse_y_test)}')
    print(f'Mean MAE Test: {np.mean(mae_y_test)}')

    # Calculate threshold and anomalies based on the given plot_metric
    if plot_metric.lower() == 'mse':
        threshold = np.percentile(mse_y_test, pourcentage)
        anomalies = mse_y_test > threshold
        metric_y_test = mse_y_test
    else:  # Default to mae
        threshold = np.percentile(mae_y_test, pourcentage)
        anomalies = mae_y_test > threshold
        metric_y_test = mae_y_test

    # Plot the histogram of the metric
    plt.hist(metric_y_test, bins=50)
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
    plt.show()
    
    anomalies_df = pd.DataFrame()
    anomalies_df['Time Step'] = np.arange(len(metric_y_test))
    anomalies_df['y_test'] = y_test.mean(axis=1).flatten()  # take the mean of y_test within each sequence
    anomalies_df[plot_metric.upper()] = metric_y_test
    anomalies_df[f'{plot_metric.upper()}_threshold'] = threshold
    anomalies_df['Is Anomaly'] = anomalies
    
    # Plot the metric over time
    plt.figure(figsize=(10,6))
    plt.plot(metric_y_test, label=plot_metric.upper())
    plt.plot([threshold]*len(metric_y_test), color='r', label='Threshold')  # Threshold line
    plt.scatter(anomalies_df[anomalies_df['Is Anomaly']]['Time Step'], anomalies_df[anomalies_df['Is Anomaly']][plot_metric.upper()], color='k', label='Anomalies')  # Anomalies
    plt.title(f'{plot_metric.upper()} Over Time')
    plt.xlabel('Time Step')
    plt.ylabel(plot_metric.upper())
    plt.legend()
    plt.show()
    
    # plot Anomalies : 
    # Plot y_test with anomalies points
    plt.figure(figsize=(10,6))
    plt.plot(anomalies_df['y_test'], label='y_test')
    plt.scatter(anomalies_df[anomalies_df['Is Anomaly']].index, anomalies_df[anomalies_df['Is Anomaly']]['y_test'], color='r', label='Anomalies')
    plt.title('y_test Over Time with Anomalies')
    plt.xlabel('Time Step')
    plt.ylabel('y_test')
    plt.legend()
    plt.show()
    
evaluate_and_plot(model, X_test, y_test, y_train, 10,'mae',99)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def evaluate_and_plot(model, data_test, y_train, sequence_length, plot_metric='mae', pourcentage=90):
    # Make predictions (reconstruct the input) on the test data
    data_test_pred = model.predict(data_test)

    # Calculate the MSE and MAE between the original and reconstructed data
    mse_data_test = np.mean(np.power(data_test_pred - data_test, 2), axis=1)
    mae_data_test = np.mean(np.abs(data_test_pred - data_test), axis=1)

    # Print the mean MSE and MAE
    print(f'Mean MSE Test: {np.mean(mse_data_test)}')
    print(f'Mean MAE Test: {np.mean(mae_data_test)}')

    # Calculate threshold and anomalies based on the given plot_metric
    if plot_metric.lower() == 'mse':
        threshold = np.percentile(mse_data_test, pourcentage)
        anomalies = mse_data_test > threshold
        metric_data_test = mse_data_test
    else:  # Default to mae
        threshold = np.percentile(mae_data_test, pourcentage)
        anomalies = mae_data_test > threshold
        metric_data_test = mae_data_test

    # Plot the histogram of the metric
    plt.hist(metric_data_test, bins=50)
    plt.axvline(x=threshold, color='r', linestyle='dashed', linewidth=2)
    plt.show()
    
    anomalies_df = pd.DataFrame()
    anomalies_df['Time Step'] = np.arange(len(metric_data_test))
    anomalies_df['data_test'] = data_test.mean(axis=1).flatten()  # take the mean of data_test within each sequence
    anomalies_df[plot_metric.upper()] = metric_data_test
    anomalies_df[f'{plot_metric.upper()}_threshold'] = threshold
    anomalies_df['Is Anomaly'] = anomalies
    
    # Plot the metric over time
    plt.figure(figsize=(10,6))
    plt.plot(metric_data_test, label=plot_metric.upper())
    plt.plot([threshold]*len(metric_data_test), color='r', label='Threshold')  # Threshold line
    plt.scatter(anomalies_df[anomalies_df['Is Anomaly']]['Time Step'], anomalies_df[anomalies_df['Is Anomaly']][plot_metric.upper()], color='k', label='Anomalies')  # Anomalies
    plt.title(f'{plot_metric.upper()} Over Time')
    plt.xlabel('Time Step')
    plt.ylabel(plot_metric.upper())
    plt.legend()
    plt.show()
    
    # Plot data_test with anomalies points
    plt.figure(figsize=(10,6))
    plt.plot(anomalies_df['data_test'], label='data_test')
    plt.scatter(anomalies_df[anomalies_df['Is Anomaly']].index, anomalies_df[anomalies_df['Is Anomaly']]['data_test'], color='r', label='Anomalies')
    plt.title('data_test Over Time with Anomalies')
    plt.xlabel('Time Step')
    plt.ylabel('data_test')
    plt.legend()
    plt.show()

evaluate_and_plot(model, X_test, y_train, 10,'mae',99) # For X_test
evaluate_and_plot(model, y_test, y_train, 10,'mae',99) # For y_test




# """_summary_
# Tuning LSTM Parameters : Using KerasTuner
# """

# from tensorflow import keras
# from tensorflow.keras import layers
# from kerastuner.tuners import RandomSearch
# def build_model(hp):
#     model = keras.Sequential()
#     model.add(layers.LSTM(units=hp.Int('units_1', min_value=10, max_value=128, step=32), 
#                           activation='relu', 
#                           input_shape=(X_train.shape[1], X_train.shape[2]), 
#                           return_sequences=True))
#     model.add(layers.LSTM(units=hp.Int('units_2', min_value=10, max_value=128, step=32), 
#                           activation='relu', 
#                           return_sequences=False))
#     model.add(layers.RepeatVector(X_train.shape[1]))
#     model.add(layers.LSTM(units=hp.Int('units_3', min_value=10, max_value=128, step=32), 
#                           activation='relu', 
#                           return_sequences=True))
#     model.add(layers.LSTM(units=hp.Int('units_4', min_value=10, max_value=128, step=32), 
#                           activation='relu', 
#                           return_sequences=True))
#     model.add(layers.TimeDistributed(layers.Dense(X_train.shape[2])))

#     model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
#                   loss='mse')

#     return model


# def tuning_model(X_train, y_train):
#     tuner = RandomSearch(
#         build_model,
#         objective='val_loss',
#         max_trials=5,
#         executions_per_trial=3,
#         directory='my_dir',
#         project_name='10_seq')

#     tuner.search_space_summary()

#     tuner.search(X_train, y_train, epochs=5, validation_split=0.1)

#     tuner.results_summary()

#     return tuner

# tuner = tuning_model(X_train, y_train)
