# Handwritten Digit Recognition using a Neural Network from Scratch (NumPy)

A fully connected **Artificial Neural Network (ANN)** implemented **from scratch using NumPy** to classify handwritten digits from the **MNIST** dataset. This project demonstrates the complete implementation of a neural network without using deep learning frameworks such as TensorFlow or PyTorch.

The network includes manual implementations of **forward propagation, backpropagation, stochastic gradient descent (SGD), mini-batch training, parameter updates, and performance visualization**.

---

## Overview

This project builds a digit classifier using only **NumPy** and fundamental machine learning concepts. Every component of the neural network is implemented manually, making it an excellent educational project for understanding how neural networks work internally.

The model is trained on the MNIST handwritten digit dataset and predicts digits from **0–9**.

---

## Features

- Neural Network implemented completely from scratch
- No TensorFlow, PyTorch, or Keras
- Manual Forward Propagation
- Manual Backpropagation
- Mini-Batch Stochastic Gradient Descent (SGD)
- Sigmoid Activation Function
- One-Hot Encoding
- Weight and Bias Initialization
- Accuracy Tracking
- Loss Tracking
- Training Curve Visualization
- Random Prediction Visualization

---

## Dataset

The project uses the **MNIST Handwritten Digits** dataset.

### Dataset Statistics

- **Training Samples:** 42,000 (Kaggle MNIST CSV)
- **Image Size:** 28 × 28 pixels
- **Input Features:** 784
- **Output Classes:** 10 (Digits 0–9)

Each row contains:

- First column → Label
- Remaining 784 columns → Pixel values

---

## Neural Network Architecture

```
Input Layer
784 Neurons
      │
      ▼
Hidden Layer
15 Neurons
(Sigmoid)
      │
      ▼
Output Layer
10 Neurons
(Sigmoid)
```

---

## Project Workflow

```
MNIST CSV Dataset
        │
        ▼
Data Loading
        │
        ▼
Shuffle Dataset
        │
        ▼
Mini-Batch Creation
        │
        ▼
Forward Propagation
        │
        ▼
Loss Calculation
        │
        ▼
Backpropagation
        │
        ▼
Gradient Computation
        │
        ▼
SGD Parameter Update
        │
        ▼
Accuracy & Loss Tracking
        │
        ▼
Training Visualization
        │
        ▼
Random Predictions
```

---

## Mathematical Components

### Forward Propagation

```
Z₁ = W₁X + B₁
A₁ = Sigmoid(Z₁)

Z₂ = W₂A₁ + B₂
A₂ = Sigmoid(Z₂)
```

---

### Sigmoid Activation

```
σ(x) = 1 / (1 + e⁻ˣ)
```

---

### Backpropagation

Gradients are computed manually using the chain rule to update:

- Weights
- Biases

using Mini-Batch Gradient Descent.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/mnist-neural-network-from-scratch.git

cd mnist-neural-network-from-scratch
```

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

---

## Training

Place the **MNIST training CSV** (`train.csv`) in the project directory and run:

```bash
python main.py
```

The model performs:

1. Load dataset
2. Shuffle training samples
3. Create mini-batches
4. Forward propagation
5. Backpropagation
6. Parameter updates using SGD
7. Accuracy calculation
8. Loss calculation
9. Plot training curves
10. Display random predictions

---

## Hyperparameters

| Parameter | Value |
|-----------|------:|
| Input Neurons | 784 |
| Hidden Neurons | 15 |
| Output Neurons | 10 |
| Activation | Sigmoid |
| Optimizer | Mini-Batch SGD |
| Learning Rate | 0.01 |
| Mini-Batch Size | 10 |
| Epochs | 30 |

---

## Output

During training, the program prints:

```text
Epoch 1 | Accuracy: 82.34% | Loss: 0.12451
Epoch 2 | Accuracy: 89.17% | Loss: 0.08743
...
Epoch 30 | Accuracy: 95.62% | Loss: 0.02118
```

It also generates:

- 📈 Accuracy vs Epoch graph
- 📉 Loss vs Epoch graph
- 🖼️ Random handwritten digit predictions with actual labels

---

## Repository Structure

```
.
├── main.py
├── train.csv
├── README.md
└── requirements.txt
```

---

## Future Improvements

- ReLU Activation Function
- Softmax Output Layer
- Cross-Entropy Loss
- Xavier/He Weight Initialization
- Momentum Optimizer
- Adam Optimizer
- L2 Regularization
- Dropout
- Learning Rate Scheduling
- Confusion Matrix
- Test Dataset Evaluation
- Model Saving & Loading

---

## Learning Outcomes

This project demonstrates:

- Neural Networks from Scratch
- Matrix-Based Forward Propagation
- Manual Backpropagation
- Gradient Descent Optimization
- Mini-Batch Training
- Weight Initialization
- Loss Computation
- Image Classification
- NumPy for Machine Learning
- Fundamental Deep Learning Concepts

---

## Acknowledgements

- **MNIST Handwritten Digits Dataset**
- **NumPy**
- **Pandas**
- **Matplotlib**

---

## Author

**Shivansh Kumar Sahu**

If you found this project useful or learned something from it, consider giving the repository a ⭐ on GitHub!
