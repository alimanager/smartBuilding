import pandas as pd

def get_data():
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
            merged_df = pd.merge(merged_df, df, on='timestamp [dd/mm/yyyy HH:MM]', how='outer')
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

    return merged_df

# Call the function to get the merged DataFrame
result_df = get_data()
print(result_df)
