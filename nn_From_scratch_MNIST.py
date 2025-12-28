import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Load data
data = pd.read_csv('/home/fornax/Documents/Programs/train.csv')
data = np.array(data)
print(data.shape)


# ---------------------------
# INITIALIZATION
# ---------------------------
def parameters(x, h, y):
    b1 = np.random.randn(h, 1) * 0.01
    b2 = np.random.randn(y, 1) * 0.01
    w1 = np.random.randn(h, x) * 0.01
    w2 = np.random.randn(y, h) * 0.01
    return b1, b2, w1, w2


# ---------------------------
# ACTIVATION
# ---------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------
# SHUFFLING + MINI BATCHING
# ---------------------------
def data_distro(data, size):
    np.random.shuffle(data)
    mini_batches = [data[k:k+size] for k in range(0, len(data), size)]
    return mini_batches


# ---------------------------
# FEED FORWARD
# ---------------------------
def feed_forward(x, w1, w2, b1, b2):
    z1 = w1 @ x + b1
    a1 = sigmoid(z1)
    z2 = w2 @ a1 + b2
    a2 = sigmoid(z2)
    return z1, z2, a1, a2


# ---------------------------
# ONE-HOT
# ---------------------------
def one_hot(true_value):
    y = np.zeros((10, 1))
    y[true_value] = 1
    return y


# ---------------------------
# BACKPROP
# ---------------------------
def back_prop(w1, w2, a1, a2, z1, z2, y, x):

    dz2 = (a2 - y) * (a2 * (1 - a2))
    dw2 = dz2 @ a1.T
    db2 = dz2

    dz1 = (w2.T @ dz2) * (a1 * (1 - a1))
    dw1 = dz1 @ x.T
    db1 = dz1

    return dw1, dw2, db1, db2


# ---------------------------
# UPDATE
# ---------------------------
def update_parameters(w1, w2, b1, b2, dw1, dw2, db1, db2, alpha):
    w1 -= alpha * dw1
    w2 -= alpha * dw2
    b1 -= alpha * db1
    b2 -= alpha * db2
    return w1, w2, b1, b2


# ---------------------------
# ACCURACY
# ---------------------------
def get_accuracy(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    return np.sum(pred == true) / pred.size


# ---------------------------
# SGD + GRAPHS + RANDOM OUTPUTS
# ---------------------------
def SGD(data, epochs, mini_batch_size, alpha):

    b1, b2, w1, w2 = parameters(784, 15, 10)

    accuracy_history = []
    loss_history = []

    for epoch in range(epochs):
        mini_batches = data_distro(data, mini_batch_size)

        all_preds = []
        all_true = []
        total_cost = 0
        total_items = 0

        for mini_batch in mini_batches:

            dw1_sum = np.zeros_like(w1)
            dw2_sum = np.zeros_like(w2)
            db1_sum = np.zeros_like(b1)
            db2_sum = np.zeros_like(b2)

            for image in mini_batch:
                true_value = int(image[0])
                x = image[1:].astype(np.float32).reshape(-1, 1) / 255.0

                z1, z2, a1, a2 = feed_forward(x, w1, w2, b1, b2)
                pred = np.argmax(a2)

                all_preds.append(pred)
                all_true.append(true_value)

                y = one_hot(true_value)
                total_cost += np.sum((a2 - y) ** 2) / 10
                total_items += 1

                dw1, dw2, db1, db2 = back_prop(w1, w2, a1, a2, z1, z2, y, x)

                dw1_sum += dw1
                dw2_sum += dw2
                db1_sum += db1
                db2_sum += db2

            w1, w2, b1, b2 = update_parameters(
                w1, w2, b1, b2,
                dw1_sum/mini_batch_size,
                dw2_sum/mini_batch_size,
                db1_sum/mini_batch_size,
                db2_sum/mini_batch_size,
                alpha
            )

        acc = get_accuracy(all_preds, all_true) * 100
        loss = total_cost / total_items

        accuracy_history.append(acc)
        loss_history.append(loss)

        print(f"Epoch {epoch+1} | Accuracy: {acc:.2f}% | Loss: {loss:.5f}")

    # ---------------- PLOTTING ----------------
    plt.figure(figsize=(14,5))

    plt.subplot(1,2,1)
    plt.plot(accuracy_history, marker='o')
    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.subplot(1,2,2)
    plt.plot(loss_history, marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()

    # ---------------- SHOW RANDOM PREDICTIONS ----------------
    print("\nShowing random predictions:\n")

    for _ in range(5):
        idx = random.randint(0, len(data) - 1)
        label = int(data[idx][0])
        img = data[idx][1:].reshape(28,28)

        x = data[idx][1:].reshape(-1,1) / 255.0
        _, _, _, out = feed_forward(x, w1, w2, b1, b2)
        pred = np.argmax(out)

        plt.imshow(img, cmap='gray')
        plt.title(f"Predicted: {pred} | Actual: {label}")
        plt.show()


# Run training
SGD(data, epochs=30, mini_batch_size=10, alpha=0.01)
