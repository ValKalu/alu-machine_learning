import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load data
data = pd.read_csv('data/train/training_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP model
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, predictions))

# Save the model
pickle.dump(model, open('../models/talent_success_model.pkl', 'wb'))
import pandas as pd
import pickle

# Load processed data
data_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/train/'
artists_df = pd.read_csv(data_path + 'artists.csv')
tracks_df = pd.read_csv(data_path + 'tracks.csv')
engagement_df = pd.read_csv(data_path + 'engagement.csv')
revenue_df = pd.read_csv(data_path + 'revenue.csv')

# Placeholder: example processing
# Example: merging data (if needed)
merged_df = pd.merge(tracks_df, artists_df, on='artist_id')
merged_df = pd.merge(merged_df, engagement_df, on='track_id')
merged_df = pd.merge(merged_df, revenue_df, on='track_id')

# Example: saving a processed file
processed_data_path = '/home/kalu/alu-machine_learning/summatives/talentai/data/processed/processed_data.csv'
merged_df.to_csv(processed_data_path, index=False)

# Example: training model or analysis
# Here you can add any model training or data analysis code

print("Data processing and model training completed.")
