import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data
hours = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
marks = np.array([30, 40, 50, 60, 70])

# Model
model = LinearRegression()
model.fit(hours, marks)

# Prediction
predicted = model.predict([[6]])

print("Predicted marks for 6 hours study:", predicted[0])

# Graph (to make it look advanced)
plt.scatter(hours, marks)
plt.plot(hours, model.predict(hours))
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()