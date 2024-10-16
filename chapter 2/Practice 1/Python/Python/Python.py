import numpy as np
import matplotlib.pyplot as plt

hours = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
scores = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

hours_norm = (hours - hours.mean()) / hours.std()

m = 0  
b = 0  
learning_rate = 0.001  
iterations = 20000  

def compute_cost(m, b, hours, scores):
    total_samples = len(scores)
    predictions = m * hours + b
    cost = (1 / (2 * total_samples)) * np.sum((predictions - scores) ** 2)
    return cost
def gradient_descent(hours, scores, m, b, learning_rate, iterations):
    total_samples = len(scores)
    cost_history = []

    for i in range(iterations):
        predictions = m * hours + b
  
        m_gradient = -(2 / total_samples) * np.sum(hours * (scores - predictions))
        b_gradient = -(2 / total_samples) * np.sum(scores - predictions)

        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
     
        cost = compute_cost(m, b, hours, scores)
        cost_history.append(cost)

    return m, b, cost_history

m_optimal, b_optimal, cost_history = gradient_descent(hours_norm, scores, m, b, learning_rate, iterations)

plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

predictions = m_optimal * hours_norm + b_optimal

plt.scatter(hours, scores, color='blue', label="Original Data")
plt.plot(hours, predictions, color='red', label="Fitted Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()

new_hours = np.array([170])
new_hours_norm = (new_hours - hours.mean()) / hours.std()
predicted_scores = m_optimal * new_hours_norm + b_optimal
print("Predicted Scores for new hours:", predicted_scores)