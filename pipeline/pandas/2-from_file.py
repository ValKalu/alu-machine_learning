#!/usr/bin/env python3
import pandas as pd

def from_file(filename, delimiter):
    """Loads data from a file as a pd.DataFrame.

    Args:
        filename: The file to load from.
        delimiter: The column separator.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    # Load the data into a DataFrame
    df = pd.read_csv(filename, delimiter=delimiter)
    return df

# This is for testing the script directly
if __name__ == "__main__":
    df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
    print(df1.head())
    df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')
    print(df2.tail())
