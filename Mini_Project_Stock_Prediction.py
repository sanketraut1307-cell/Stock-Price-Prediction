# Simple Stock Price Prediction
# Author: Sanket Raut
# Language: Python
# Libraries used: pandas, matplotlib, sklearn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Create sample stock data
data = {
    "Day": [1, 2, 3, 4, 5, 6, 7],
    "Price": [100, 102, 101, 105, 107, 110, 115]
}

df = pd.DataFrame(data)
print(df)

# Step 2: Prepare data for training
X = df[["Day"]]       
y = df["Price"]       

# Step 3: Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict future prices (e.g., next 3 days)
fut_days = pd.DataFrame({"Day": [8, 9, 10]})
pre_prices = model.predict(fut_days)

# Step 5: Show results
print("\nPredicted Prices:")
for d, p in zip(fut_days["Day"], pre_prices):
    print(f"Day {d}: {p:.2f}")
print()

# Step 6: Plot actual and predicted prices
plt.plot(df["Day"], df["Price"], label="Actual Price", marker='o', markersize=8, linewidth=2, color="#28282A", markerfacecolor="#FFFFFF")
plt.plot(fut_days["Day"], pre_prices, label="Predicted Price", marker='x',markersize=8, linewidth=2, linestyle='dashdot', color="#6A5608")
plt.xlabel("Day",fontsize=15
                ,family="Arial"
                ,color="#000000F6")
plt.ylabel("Stock Price",fontsize=15
                        ,family="Arial"
                        ,color="#000000F6")
plt.title("Simple Stock Price Prediction", fontsize=15)
plt.grid(linewidth=1.5, color="lightgray", linestyle="dashed")
plt.legend()
plt.show()
