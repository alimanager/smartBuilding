import pandas as pd 
import kaggle
import matplotlib.pyplot as plt



# Replace 'dataset-name' with the actual dataset name from Kaggle
dataset_name = 'claytonmiller/occupant-presence-and-actions-in-office-building'

# Download the dataset to the current directory
kaggle.api.dataset_download_files(dataset_name, unzip=True)

# Replace 'dataset_filename.csv' with the actual filename of your dataset
df = pd.read_csv('01_occ.csv')
# Display the first few rows of the dataset
print(df.head())

# Get a summary of the dataset (data types, missing values, etc.)
print(df.info())


# Descriptive statistics for numerical columns
# print(df.describe())

# Convert the timestamp column to a proper datetime format (assuming the column name is 'timestamp')
df['timestamp'] = pd.to_datetime(df['timestamp [dd/mm/yyyy HH:MM]'])


import os
import pandas as pd

# Step 1: Get a list of all CSV files in your environment
csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dataframes = []

# Step 2: Read each CSV file into a DataFrame and store it in the list
for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

# Step 3: Merge the DataFrames based on the 'timestamp' column using pd.concat
merged_df = pd.concat(dataframes, ignore_index=True)

# Display the merged DataFrame
print(merged_df)



# ## Occupancy rate : 
# df.loc[:, 1:].sum()
# # Calculate the total time (in minutes) for each area (column)
# total_time_by_area = df.iloc[:, 1:].mean() 

# binary_columns = df.columns[df.nunique() == 2]  #
# rates = {}  # Dictionary to store rates for each binary column

# for column in binary_columns:
#     rate = df[column].mean()
#     rates[column] = rate

# # Output the rates
# for column, rate in rates.items():
#     print(f"Rate of '1' in '{column}' column: {rate:.2f}")
