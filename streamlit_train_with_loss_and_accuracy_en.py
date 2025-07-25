
import streamlit as st
import numpy as np
import struct
import matplotlib.pyplot as plt

# Load MNIST data in IDX format
@st.cache_data
def load_images(file_path):
    with open(file_path, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((num, rows * cols)) / 255.0

@st.cache_data
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

# Load data (make sure files are in the same directory)
train_images = load_images("train-images.idx3-ubyte")
train_labels = load_labels("train-labels.idx1-ubyte")
test_images = load_images("t10k-images.idx3-ubyte")
test_labels = load_labels("t10k-labels.idx1-ubyte")

# Activation and utility functions
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return x > 0
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    m = y_pred.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true] + 1e-9)
    return np.mean(log_likelihood)

def one_hot(y, num_classes): return np.eye(num_classes)[y]

# Streamlit UI
st.title("🧠 Neural Network - Loss Curve and Accuracy")

epochs = st.slider("Number of Epochs", 1, 100, 10)
lr = st.slider("Learning Rate", 0.001, 1.0, 0.01)

if st.button("Start Training"):
    input_size = 784
    hidden_size = 64
    output_size = 10

    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / (input_size + hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / (hidden_size + output_size))
    b2 = np.zeros((1, output_size))

    losses = []

    for epoch in range(epochs):
        z1 = np.dot(train_images, W1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = softmax(z2)

        loss = cross_entropy(y_pred, train_labels)
        losses.append(loss)

        y_true = one_hot(train_labels, output_size)
        dz2 = (y_pred - y_true) / train_images.shape[0]
        dW2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * relu_derivative(z1)
        dW1 = np.dot(train_images.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    # Test evaluation
    z1_test = np.dot(test_images, W1) + b1
    a1_test = relu(z1_test)
    z2_test = np.dot(a1_test, W2) + b2
    y_test_pred = np.argmax(softmax(z2_test), axis=1)

    accuracy = np.mean(y_test_pred == test_labels) * 100
    st.subheader(f"✅ Test Accuracy: {accuracy:.2f}%")

    # Loss curve
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, epochs + 1), losses, marker='o')
    ax1.set_title("Loss Curve During Training")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    st.pyplot(fig1)

    # Display first 10 test images
    fig2, axes = plt.subplots(2, 5, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        img = test_images[i].reshape(28, 28)
        ax.imshow(img, cmap="gray")
        ax.set_title(f"True: {test_labels[i]}\nPred: {y_test_pred[i]}")
        ax.axis('off')
    st.pyplot(fig2)
