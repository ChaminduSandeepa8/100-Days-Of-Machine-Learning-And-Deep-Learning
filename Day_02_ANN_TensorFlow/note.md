# Day 02: Deep Neural Networks with TensorFlow & Keras

Today, I transitioned from building Neural Networks from scratch with NumPy to using the powerful **TensorFlow & Keras** framework. I built and trained a multi-layer Deep Neural Network (DNN) to classify the MNIST dataset.

## 🧠 Key Learnings
* **The Sequential API:** Used `keras.Sequential` to easily stack multiple layers.
* **Deep Architecture:** Built a deep network with 5 hidden layers (512, 256, 128, 64, 32 neurons) using the `ReLU` activation function to capture complex patterns.
* **Softmax & Loss Function:** Used `Softmax` in the output layer for probability distribution across the 10 digit classes. Crucially, I used `SparseCategoricalCrossentropy` as the loss function, which automatically handles integer labels (no need to manually One-Hot encode like yesterday!).
* **Adam Optimizer:** Upgraded from basic Gradient Descent to the `Adam` optimizer for faster and more efficient convergence.
* **Model Evaluation & Visualization:** Tracked the training process using the `history` object and plotted the accuracy and loss curves using Matplotlib.

## 💻 Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

# 1. Load Data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x_train.shape)

# 2. Flatten Data (28x28 -> 784)
x_train_flatton = x_train.reshape((60000, -1))
x_test_flatton = x_test.reshape((10000, -1))

# 3. Build the Deep Neural Network
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(784,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation="softmax"),
])

# 4. Compile the Model
model.compile(
    optimizer='adam',
    loss="SparseCategoricalCrossentropy",
    metrics=['accuracy']
)

model.summary()

# 5. Train the Model
history = model.fit(x_train_flatton, y_train, epochs=5)

# 6. Evaluate
model.evaluate(x_test_flatton, y_test)

# 7. Visualize Accuracy and Loss
print(history.history.keys())
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['loss'], label='Loss')
plt.legend()
plt.show()