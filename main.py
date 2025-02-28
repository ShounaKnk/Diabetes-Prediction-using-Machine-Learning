import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
dataset = pd.read_csv('diabetes.csv')

# Separate features (X) and target variable (Y)
X = dataset.drop(columns='Outcome').to_numpy()  # Convert to NumPy array to avoid feature name issues
Y = dataset['Outcome'].to_numpy()

# Standardize the feature data
scaler = StandardScaler()
X = scaler.fit_transform(X)  # Fit & transform in one step

# Split dataset into training and testing sets (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Train the SVM model
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Model evaluation
train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
test_accuracy = accuracy_score(y_test, classifier.predict(X_test))

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Predictive system function
def predict_diabetes(input_data):
    """Predicts if a patient is diabetic (1) or not (0) based on input features."""
    input_data = np.array(input_data).reshape(1, -1)  # Convert to NumPy array and reshape
    standardized_data = scaler.transform(input_data)  # Standardize input
    prediction = classifier.predict(standardized_data)[0]  # Get prediction
    return "Patient is Diabetic" if prediction == 1 else "Patient is NOT Diabetic"

joblib.dump(classifier, "diabetes_model.pkl")
joblib.dump(scaler, 'scaler.pkl')

# Example test prediction
sample_data = (1, 115, 70, 30, 96, 34.6, 0.529, 32)
print("\nPrediction Result:", predict_diabetes(sample_data))
