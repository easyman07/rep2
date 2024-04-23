import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape((X_train.shape[0], -1)) / 255.0
X_test = X_test.reshape((X_test.shape[0], -1)) / 255.0

# One-hot encode the target labels
num_classes = 10
y_train_one_hot = np.eye(num_classes)[y_train]
y_test_one_hot = np.eye(num_classes)[y_test]

# Define the softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Define the gradient of the cross-entropy loss function with respect to the logits
def grad_cross_entropy_loss(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# Define the neural network model
input_size = X_train.shape[1]
hidden_size = 128
output_size = num_classes
learning_rate = 0.1
num_epochs = 5
batch_size = 32

# Initialize weights and biases
np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)

# Training loop
for epoch in range(num_epochs):
    # Shuffle training data
    permutation = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train_one_hot[permutation]

    for i in range(0, X_train.shape[0], batch_size):
        # Mini-batch
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]

        # Forward pass
        hidden_layer = np.maximum(0, np.dot(X_batch, W1) + b1)  # ReLU activation
        logits = np.dot(hidden_layer, W2) + b2
        y_pred = softmax(logits)

        # Compute loss
        loss = cross_entropy_loss(y_batch, y_pred)

        # Backward pass
        dlogits = grad_cross_entropy_loss(y_batch, y_pred)
        dW2 = np.dot(hidden_layer.T, dlogits)
        db2 = np.sum(dlogits, axis=0)
        dhidden = np.dot(dlogits, W2.T)
        dhidden[hidden_layer <= 0] = 0  # ReLU gradient
        dW1 = np.dot(X_batch.T, dhidden)
        db1 = np.sum(dhidden, axis=0)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')

# Evaluate the model
hidden_layer = np.maximum(0, np.dot(X_test, W1) + b1)
logits = np.dot(hidden_layer, W2) + b2
y_pred = softmax(logits)
test_loss = cross_entropy_loss(y_test_one_hot, y_pred)
predicted_labels = np.argmax(y_pred, axis=1)
accuracy = np.mean(predicted_labels == y_test)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
