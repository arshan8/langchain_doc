import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_features = X_train_scaled.shape[1]
num_classes = len(np.unique(y_train))
num_hidden = 10

W1 = np.random.randn(num_features, num_hidden) * 0.01
b1 = np.zeros((1, num_hidden))
W2 = np.random.randn(num_hidden, num_classes) * 0.01
b2 = np.zeros((1, num_classes))

epochs = 10000
learning_rate = 0.01

loss_history = []
accuracy_history = []
test_accuracy_history = []

for epoch in range(epochs):
    hidden_layer_input = np.dot(X_train_scaled, W1) + b1
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predictions = softmax(output_layer_input)

    y_one_hot = np.zeros((len(y_train), num_classes))
    y_one_hot[np.arange(len(y_train)), y_train] = 1

    loss = -np.sum(y_one_hot * np.log(predictions + 1e-9)) / len(y_train)
    loss_history.append(loss)

    train_predicted_classes = np.argmax(predictions, axis=1)
    train_accuracy = np.mean(train_predicted_classes == y_train)
    accuracy_history.append(train_accuracy)

    error_output = predictions - y_one_hot
    
    grad_W2 = np.dot(hidden_layer_output.T, error_output)
    grad_b2 = np.sum(error_output, axis=0, keepdims=True)

    error_hidden = np.dot(error_output, W2.T) * relu_derivative(hidden_layer_input)
    
    grad_W1 = np.dot(X_train_scaled.T, error_hidden)
    grad_b1 = np.sum(error_hidden, axis=0, keepdims=True)

    W2 -= learning_rate * grad_W2
    b2 -= learning_rate * grad_b2
    W1 -= learning_rate * grad_W1
    b1 -= learning_rate * grad_b1

    hidden_layer_input_test = np.dot(X_test_scaled, W1) + b1
    hidden_layer_output_test = relu(hidden_layer_input_test)
    output_layer_input_test = np.dot(hidden_layer_output_test, W2) + b2
    test_predictions = softmax(output_layer_input_test)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    test_accuracy = np.mean(test_predicted_classes == y_test)
    test_accuracy_history.append(test_accuracy)

    if epoch % (epochs // 10) == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Train Accuracy')
plt.plot(test_accuracy_history, label='Test Accuracy')





import numpy as np
from sklearn.utils.extmath import approx_fprime

# Define scalar function f: ℝⁿ → ℝ
def f(x):
    return np.sum(x**2 + 3*x)  # Example: f(x, y) = x² + y² + 3x + 3y

# Gradient = Jacobian for scalar-valued functions
def compute_jacobian(f, x, epsilon=1e-6):
    return approx_fprime(x, f, epsilon)

# Numerical Hessian
def compute_hessian(f, x, epsilon=1e-5):
    n = len(x)
    hessian = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x_i1 = x.copy()
        x_i1[i] += epsilon
        f_i1 = f(x_i1)

        for j in range(n):
            x_ij = x_i1.copy()
            x_ij[j] += epsilon
            f_ij = f(x_ij)

            x_j1 = x.copy()
            x_j1[j] += epsilon
            f_j1 = f(x_j1)

            hessian[i, j] = (f_ij - f_i1 - f_j1 + fx) / (epsilon ** 2)
    return hessian

# Test input
x0 = np.array([1.0, 2.0])

# Compute
jac = compute_jacobian(f, x0)
hess = compute_hessian(f, x0)

print("Jacobian (Gradient):", jac)
print("Hessian:\n", hess)

plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

final_test_accuracy = np.mean(predicted_classes == y_test)
print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
