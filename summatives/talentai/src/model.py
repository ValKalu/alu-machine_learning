import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from preprocessing import preprocess_data

# Load and preprocess data
X, y = preprocess_data('../data/train/train_data.csv')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, '../models/talentai_model.pkl')
