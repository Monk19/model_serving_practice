import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train a very simple model
model = LinearRegression()
model.fit(X, y)

# Save it
joblib.dump(model, 'app/model.pkl')
print("Model trained and saved to app/model.pkl")