import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Download and prepare the data

data_root = "https://github.com/ageron/data/raw/main/"
dataset = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = dataset[["GDP per capita (USD)"]].values
y = dataset[["Life satisfaction"]].values

# Select a linear model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# Train the model
model.fit(X, y)
# Make a prediction for Cyprus
X_new = [[37655.2]] 
print(model.predict(X_new)) # output: 6.30165767

y_pred = model.predict(X)
# Visualize the data
dataset.plot(kind='scatter', grid=True,x="GDP per capita (USD)", y="Life satisfaction")
plt.plot(X, y_pred, color='red', label='Regression line')
plt.show()
