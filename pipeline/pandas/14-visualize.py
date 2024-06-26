#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove the Weighted_Price column
df = df.drop(columns=['Weighted_Price'])

# Rename the Timestamp column to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Set the Date column as the index
df = df.set_index('Date')

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 and beyond
df = df.loc['2017-01-01':]

# Resample data at daily intervals and aggregate
daily_df = df.resample('D').agg({
    'Open': 'mean',
    'High': 'max',
    'Low': 'min',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(daily_df.index, daily_df['Close'], label='Close')
plt.title('Daily Close Prices (2017 and beyond)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
