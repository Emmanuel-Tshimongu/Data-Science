import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
# Generate synthetic data
np.random.seed(42)  # For reproducibility
# Create a DataFrame with 1000 samples and 10 features
data = {
    'age': np.random.randint(1, 15, 1000),  # Age of the car (1 to 15 years)
    'mileage': np.random.randint(10000, 200000, 1000),  # Mileage in km
    'engine_size': np.random.uniform(1.0, 4.0, 1000),  # Engine size in liters
    'horsepower': np.random.randint(70, 300, 1000),  # Horsepower
    'doors': np.random.choice([2, 4], 1000),  # Number of doors
    'seats': np.random.choice([2, 4, 5, 7], 1000),  # Number of seats
    'brand': np.random.choice(['Toyota', 'Ford', 'BMW', 'Audi', 'Honda'], 1000),  # Car brand
    'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric'], 1000),  # Fuel type
    'transmission': np.random.choice(['Manual', 'Automatic'], 1000),  # Transmission type
    'sunroof': np.random.choice([0, 1], 1000),  # Whether it has a sunroof (binary)
    'price': np.random.randint(5000, 50000, 1000)  # Price of the car (target variable)
}
df = pd.DataFrame(data)
# Encode categorical variables using LabelEncoder
le_brand = LabelEncoder()
df['brand'] = le_brand.fit_transform(df['brand'])
le_fuel = LabelEncoder()
df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])
le_transmission = LabelEncoder()
df['transmission'] = le_transmission.fit_transform(df['transmission'])
# Define the features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# You can check the predicted prices
print(f"First 10 predicted prices: {y_pred[:10]}")
