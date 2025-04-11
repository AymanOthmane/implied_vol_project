import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import io
import yfinance as yf

def read_options_csv(file_path = r'./spx_quotes.csv', table_start_keyword = 'Expiration Date' ):
    """
    Reads a CSV file, separates metadata from the main data table, and returns them as DataFrames.

    Args:
        file_path (str): The path to the CSV file.
        table_start_keyword (str): A keyword indicating the start of the main data table.

    Returns:
        tuple: A tuple containing two DataFrames: one for metadata and one for the main data table.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the index where the table starts
    table_start_index = None
    for i, line in enumerate(lines):
        if table_start_keyword in line:
            table_start_index = i  # Start reading data 
            break

    if table_start_index is None:
        raise ValueError("Table start keyword not found in the file.")

    # Read metadata (lines before the table starts)
    metadata_lines = lines[:table_start_index]

    # Clean and parse metadata
    metadata_dict = parse_metadata(metadata_lines)

    metadata_df = pd.DataFrame(list(metadata_dict.values()), index=metadata_dict.keys(), columns=['Value'])
    metadata_df.loc['Date','Value'] = metadata_df.loc['Date','Value'] + ', 2024'
    metadata_df = metadata_df.drop('2024 at 6')

    # Read the main data table into a DataFrame
    data_lines = lines[table_start_index:]
    data_csv = "\n".join(data_lines)
    data_df = pd.read_csv(io.StringIO(data_csv), index_col=0)

    return metadata_df, data_df

def parse_metadata(metadata_lines):
    """
    Parses metadata lines into a dictionary of key-value pairs.

    Args:
        metadata_lines (list of str): List of lines containing metadata information.

    Returns:
        dict: A dictionary containing parsed metadata.
    """
    metadata_dict = {}
    for line in metadata_lines:
        # Remove newline characters and split by commas
        line = line.strip().replace('\n', '').replace('"', '')
        parts = line.split(',')

        # Iterate over parts to create key-value pairs
        for part in parts:
            if ':' in part:
                key, value = map(str.strip, part.split(':', 1))
                metadata_dict[key] = value

    return metadata_dict

def get_aod_close(date):
    """
    Get SPX last close for specific date.

    Returns:
        float: SPX last close for date
    """
    return yf.download('^SPX',start=date, end=date + relativedelta(day=1))['Close'].loc[:,'^SPX'].iloc[0]

def get_hist_vol(date):
    """
    Get one year historical volatility of S&P 500 through yfinance or assume 0.2 as first guess for IV algorithme.

    Returns:
        float: one year historical volatility(annualized) of S&P 500 or 0.20.
    """
    try:
        spx = yf.download('^SPX', start= date - relativedelta(years=1), end= date)['Close']
        return (np.log(spx / spx.shift(1)).dropna().std() * np.sqrt(252)).iloc[0]
    except (KeyError, IndexError, ValueError, RuntimeError) as e:
        print(f"An error occurred: {e}")
        return 0.2  
    
def date_difference_in_years(date1, date2):
    """
    Calculate the difference between two dates in years.

    Args:
        date1 (str or datetime): The first date (can be a string in 'YYYY-MM-DD' format or a datetime object).
        date2 (str or datetime): The second date (can be a string in 'YYYY-MM-DD' format or a datetime object).

    Returns:
        float: The difference between the two dates in years.
    """
    # Convert input dates to datetime objects if they are strings
    if isinstance(date1, str):
        date1 = datetime.strptime(date1, '%Y-%m-%d')
    if isinstance(date2, str):
        date2 = datetime.strptime(date2, '%Y-%m-%d')

    # Calculate the difference using relativedelta
    delta = relativedelta(date2, date1)

    # Extract the difference in years
    difference_in_years = delta.years + delta.months / 12 + delta.days / 365.25

    return difference_in_years
