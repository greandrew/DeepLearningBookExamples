import requests
import zipfile
import os
import pandas as pd
import glob

def download_and_extract_zip(url, extract_to='./extracted_data'):
    # Ensure the extraction directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the zip file
    response = requests.get(url, stream=True)
    zip_path = os.path.join(extract_to, url.split('/')[-1])

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Optionally, remove the downloaded zip file after extraction
    os.remove(zip_path)

def load_stockdata_from_directory(directory):
    # List all files matching the pattern YYYY-MM-DDstocks.csv
    files = glob.glob(os.path.join(directory, '*stocks.csv'))

    dataframes = []

    for file in files:
        # Extract date from filename
        date_str = os.path.basename(file).split('stocks.csv')[0]

        # Load csv file into dataframe
        df = pd.read_csv(file)
        
        # Add date column to dataframe
        df['Date'] = date_str
        
        dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df

def load_optiondata_from_directory(directory):
    # List all files matching the pattern YYYY-MM-DDoptions.csv
    files = glob.glob(os.path.join(directory, '*options.csv'))

    dataframes = []

    for file in files:
        # Load csv file into dataframe
        df = pd.read_csv(file)
        
        dataframes.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    return combined_df
