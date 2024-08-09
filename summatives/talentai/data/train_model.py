import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tensorflow as tf
from tensorflow import keras

# Define paths
data_path = './talentai_artist_records.csv'  # Path to your dataset
preprocessor_path = './pipeline/preprocessor.pkl'
model_path = './models/model.h5'

# Load dataset
data = pd.read_csv(data_path)

# Define feature columns and target column
numeric_features = ['popularity', 'followers', 'monthly_listeners']  # Numeric features from your dataset
categorical_features = ['genre', 'location', 'label']  # Categorical features from your dataset
target_column = 'signed_to_label'  # Target column representing the artist's label signing status

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing for numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create preprocessing and training pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Split data into training and testing sets
X = data[numeric_features + categorical_features]
y = data[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

# Save preprocessor
joblib.dump(preprocessor, preprocessor_path)

# If using TensorFlow for model conversion, convert and save the model
tf_model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train TensorFlow model
tf_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the TensorFlow model in .h5 format
tf_model.save(model_path)
