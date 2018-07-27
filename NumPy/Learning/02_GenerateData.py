from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Make up some real data
x_data = np.linspace(0, 4 * 3.14, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.hstack(((x_data / (3.14)) * np.sin(x_data) + noise, (x_data / (3.14)) * np.cos(x_data) + noise))

print(y_data)