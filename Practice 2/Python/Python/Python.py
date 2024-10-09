import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('Practice2_Chapter2.csv')

X = np.array([data['TV'], data['Radio'], data['Newspaper']]).T
y = np.array(data['Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.column_stack((np.ones(len(X_train_scaled)), X_train_scaled))
X_test_scaled = np.column_stack((np.ones(len(X_test_scaled)), X_test_scaled))

theta = np.zeros(X_train_scaled.shape[1])
alpha = 0.01
num_iters = 1500

def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for i in range(num_iters):
        predictions = hypothesis(X, theta)

        theta = theta - (alpha / m) * np.dot(X.T, (predictions - y))

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

theta_optimal, cost_history = gradient_descent(X_train_scaled, y_train, theta, alpha, num_iters)

y_pred = hypothesis(X_test_scaled, theta_optimal)

ss_tot = np.sum((y_test - np.mean(y_test))**2)
ss_res = np.sum((y_test - y_pred)**2)
r_squared = 1 - (ss_res / ss_tot)

print("Coefficients:")
print(f"Intercept (b0): {theta_optimal[0]:.4f}")
print(f"TV (b1): {theta_optimal[1]:.4f}")
print(f"Radio (b2): {theta_optimal[2]:.4f}")
print(f"Newspaper (b3): {theta_optimal[3]:.4f}")
print(f"R-squared on the test set: {r_squared:.4f}")

new_data = np.array([[200, 50, 100]])  
new_data_scaled = scaler.transform(new_data)
new_data_scaled = np.column_stack((np.ones(len(new_data_scaled)), new_data_scaled))
predicted_sales = hypothesis(new_data_scaled, theta_optimal)
print(f"Predicted sales for advertising budget (TV: $200k, Radio: $50k, Newspaper: $100k): ${predicted_sales[0]:.2f}k")

plt.figure(figsize=(10, 6))
plt.plot(cost_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost function value')
plt.title('Cost function over iterations')
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', marker='o', alpha=0.5)
ax.set_xlabel('TV Advertising')
ax.set_ylabel('Radio Advertising')
ax.set_zlabel('Sales')
ax.set_title('3D Scatter plot: TV & Radio Advertising vs Sales')
plt.show()
