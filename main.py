import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# File paths for saved model and scaler
MODEL_FILE = 'diabetes_model.pkl'
SCALER_FILE = 'scaler.pkl'

def train_model():
    """Trains the model with updated dataset and saves it."""
    print("\nTraining the model with the latest data...\n")
    
    # Load dataset
    dataset = pd.read_csv('diabetes.csv')

    # Separate features (X) and target variable (Y)
    X = dataset.drop(columns='Outcome').to_numpy()  # Convert to NumPy array
    Y = dataset['Outcome'].to_numpy()

    # Standardize the feature data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Fit & transform in one step

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

    # Train the SVM model
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # Model evaluation
    train_accuracy = accuracy_score(y_train, classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(X_test))

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Save the trained model and scaler
    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    
    print("\nModel and scaler saved successfully!\n")

def load_model():
    """Loads the saved model and scaler if they exist."""
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        print("\nNo saved model found. Please train the model first.\n")
        return None, None

    classifier = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("\nModel and scaler loaded successfully!\n")
    return classifier, scaler

test_data = (1,115,70,30,96,34.6,0.529,32)
def predict_diabetes():
    """Uses the saved model to make predictions."""
    classifier, scaler = load_model()
    if classifier is None or scaler is None:
        return  # Exit if model is not trained

    print("\nEnter patient details:")
    try:
        input_data = [
            float(input("Pregnancies: ")),
            float(input("Glucose: ")),
            float(input("Blood Pressure: ")),
            float(input("Skin Thickness: ")),
            float(input("Insulin: ")),
            float(input("BMI: ")),
            float(input("Diabetes Pedigree Function: ")),
            float(input("Age: "))
        ] or test_data
    except ValueError:
        print("\nInvalid input. Please enter numbers only.\n")
        return

    input_data = np.array(input_data).reshape(1, -1)
    standardized_data = scaler.transform(input_data)  # Use the saved scaler
    prediction = classifier.predict(standardized_data)[0]  # Use the saved model

    result = "Patient is Diabetic" if prediction == 1 else "Patient is NOT Diabetic"
    print("\nPrediction Result:", result, "\n")

# Menu system
while True:
    print("\n=== Diabetes Prediction System ===")
    print("1. Train the Model with Updated Data")
    print("2. Use the Model to Predict Diabetes")
    print("0. Exit")

    choice = input("\nEnter your choice: ")

    if choice == '1':
        train_model()
    elif choice == '2':
        predict_diabetes()
    elif choice == '0':
        break
    else:
        print("\nInvalid choice\n")
