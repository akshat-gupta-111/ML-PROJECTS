# backend.py
import joblib
import numpy as np

# Load model using joblib
model = joblib.load('iris-data-prediction.pkl')  # or .joblib if that's your extension

print(model.predict([[23, 34,53,33]]))
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    return prediction[0]