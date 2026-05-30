# Day 01: Artificial Neural Networks (ANN) - Initial Setup

Today I started my 100 Days of ML journey by diving into ANNs! I am currently building a basic Gradient Descent algorithm from scratch using NumPy to classify the MNIST dataset.

## 🧠 Progress So Far
* **Data Preprocessing:** Loaded the MNIST dataset and normalized pixel values (scaling between 0 and 1).
* **Flattening Data:** Reshaped the 28x28 image matrices into 1D vectors of 784 features (`x_train_flatton` and `x_test_flatton`).
* **One-Hot Encoding:** Converted the output labels into 10 categories using `tf.keras.utils.to_categorical`.
* **Gradient Descent Setup:** Initialized the Weights (`w`) and Biases (`b`) with zeros.
* **Forward Pass & Error:** Calculated the predicted output using matrix multiplication (`np.dot(x_train, w) + b`) and found the initial error margin.

## 💻 Python Implementation (Work in Progress)

```python
import numpy as np
import matplotlib.pyplot as plt
#%%
import tensorflow as tf
from tensorflow import keras
#%%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
#%%
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
#%%
plt.imshow(x_train[5],cmap='gray')
#%%
x_train_flatton = x_train.reshape(60000,784)
x_test_flatton = x_test.reshape(10000, 784)
#%%
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

#%%
print(x_train_flatton.shape)
print(y_train_one_hot.shape)
print(x_test_flatton.shape)
print(y_test_one_hot.shape)
#%%
def gradient_descent(x_train, y_train, epochs, learning_rate):
    w = np.zeros((784,10))
    b = np.zeros((1,10))
    
    for i in range(epochs):
        y_pred = np.dot(x_train,w)+b
        err = y_train - y_pred
        print(y_pred.shape)
        
    
#%%
gradient_descent(x_train_flatton,y_train_one_hot,1,0.001)