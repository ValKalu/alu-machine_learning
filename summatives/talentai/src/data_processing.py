import pandas as pd
import os

# Define file paths
base_path = '/home/kalu/alu-machine_learning/summatives/talentai'
artists_path = os.path.join(base_path, 'data/train/artists.csv')
tracks_path = os.path.join(base_path, 'data/train/tracks.csv')
engagement_path = os.path.join(base_path, 'data/train/engagement.csv')
revenue_path = os.path.join(base_path, 'data/train/revenue.csv')
output_path = os.path.join(base_path, 'data/train/training_data.csv')

# Load data
artists_df = pd.read_csv(artists_path)
tracks_df = pd.read_csv(tracks_path)
engagement_df = pd.read_csv(engagement_path)
revenue_df = pd.read_csv(revenue_path)

# Merge data
merged_df = pd.merge(tracks_df, artists_df, on='artist_id')
merged_df = pd.merge(merged_df, engagement_df, on='track_id')
merged_df = pd.merge(merged_df, revenue_df, on='track_id')

# Save merged data to CSV
merged_df.to_csv(output_path, index=False)

print(f"Data successfully saved to {output_path}")
