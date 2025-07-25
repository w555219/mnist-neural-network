# mnist-neural-network
Simple neural network from scratch for MNIST

# 🧠 MNIST Neural Network from Scratch (Streamlit)

This project implements a simple neural network from scratch **using only NumPy** to classify handwritten digits from the MNIST dataset. It features an interactive **Streamlit app** for training, tuning, and visualizing performance.

---

## 📦 Features

- Neural network with **1 hidden layer** (ReLU activation)
- Output layer with **Softmax** and **Cross-Entropy** loss
- Customizable **Epochs** and **Learning Rate** from Streamlit UI
- Interactive **loss curve plot**
- Displays **first 10 test images** with predictions
- Shows **Test Accuracy** at the end

---

## 🧪 How to Run

### 1. Install dependencies

```bash
pip install streamlit matplotlib numpy
```

### 2. Download MNIST files (IDX format)

Place these 4 files in the same folder as the script:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Download from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

### 3. Run the app

```bash
streamlit run streamlit_train_with_loss_and_accuracy_en.py
```

---

## 🧠 Neural Network Architecture

- Input layer: 784 nodes (28x28 pixels)
- Hidden layer: 64 neurons (ReLU)
- Output layer: 10 neurons (Softmax)

---

## 📊 Output

- Epoch-wise loss displayed live
- Final test accuracy printed
- Loss curve visualized with Matplotlib
- Grid of test digits with predicted vs true labels

---

## 📚 Educational Purpose

This is a great project for:
- Learning how neural networks work internally
- Understanding backpropagation
- Experimenting with hyperparameters

---

## 🛠 Future Improvements

- Add model saving/loading
- Confusion matrix visualization
- Drawing pad for user digit input

---

Made with ❤️ using NumPy and Streamlit.
