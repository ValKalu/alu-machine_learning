import pandas as pd

# Define paths
artists_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/train/artists.csv'
tracks_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/train/tracks.csv'
engagement_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/train/engagement.csv'
revenue_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/train/revenue.csv'

# Load data into DataFrames
try:
    artists_df = pd.read_csv(artists_path)
    tracks_df = pd.read_csv(tracks_path)
    engagement_df = pd.read_csv(engagement_path)
    revenue_df = pd.read_csv(revenue_path)

    # Display first few rows to verify
    print("Artists:")
    print(artists_df.head())
    print("\nTracks:")
    print(tracks_df.head())
    print("\nEngagement:")
    print(engagement_df.head())
    print("\nRevenue:")
    print(revenue_df.head())

except pd.errors.EmptyDataError:
    print("One of the files is empty or contains no data.")
except pd.errors.ParserError:
    print("Error parsing data. Check the file's format.")
except FileNotFoundError as e:
    print(f"File not found: {e}")
