# Simple Stock Price Prediction
# Author: Sanket Raut
# Language: Python
# Libraries used: pandas, matplotlib, sklearn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create sample stock data (you can replace with your own CSV)
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7],
    "Price": [100, 102, 101, 105, 107, 110, 115]
}

df = pd.DataFrame(data)
print(df)

# Step 2: Prepare data for training
X = df[["Day"]]       # Feature (input)
y = df["Price"]       # Target (output)

# Step 3: Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict future prices (e.g., next 3 days)
future_days = pd.DataFrame({"Day": [8, 9, 10]})
predicted_prices = model.predict(future_days)

# Step 5: Show results
print("\nPredicted Prices:")
for d, p in zip(future_days["Day"], predicted_prices):
    print(f"Day {d}: {p:.2f}")

# Step 6: Plot actual and predicted prices
plt.plot(df["Day"], df["Price"], label="Actual Price", marker='o')
plt.plot(future_days["Day"], predicted_prices, label="Predicted Price", marker='x', linestyle='--')
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.title("Simple Stock Price Prediction")
plt.legend()
plt.show()
