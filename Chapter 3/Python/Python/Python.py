import numpy as np
import matplotlib.pyplot as plt

# 1. Data
X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 
              0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 
              0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])

y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 
              0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 
              1, 1, 1, 1, 1, 1, 1, 1])

# 2. Initialize parameters theta and learning rate
np.random.seed(0)
theta0 = np.random.rand()
theta1 = np.random.rand()
learning_rate = 1e-4
iterations = 100

# 3. Logistic function
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# 4. Prediction function
def predict(X, theta0, theta1):
    z = theta0 + theta1 * X
    gz = logistic_function(z)
    return gz

# 5. Cost function
def cost_function(X, y_true, theta0, theta1):
    m = len(X)
    epsilon = 1e-15
    y_pred = predict(X, theta0, theta1)
    cost = -(1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cost

# 6. Gradient Descent algorithm
def gradient_descent(X, y, theta0, theta1, learning_rate):
    m = len(X)
    gradient0 = (1 / m) * np.sum(predict(X, theta0, theta1) - y)
    gradient1 = (1 / m) * np.sum((predict(X, theta0, theta1) - y) * X)
    
    theta0 = theta0 - learning_rate * gradient0
    theta1 = theta1 - learning_rate * gradient1
    
    return theta0, theta1

# 7. Train the model
for i in range(iterations):
    theta0, theta1 = gradient_descent(X, y, theta0, theta1, learning_rate)
    cost = cost_function(X, y, theta0, theta1)
    print(f"Iteration {i + 1}: Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}")

# 8. Predict results after training
y_pred = predict(X, theta0, theta1)
print(f"Predictions: {y_pred}")

# 9. Plot the prediction graph
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Prediction Line')
plt.xlabel('Sand Grain Size (mm)')
plt.ylabel('Probability of Spider Presence')
plt.title('Logistic Regression - Predicted Probability Based on Sand Grain Size')
plt.legend()
plt.grid(True)
plt.show()
