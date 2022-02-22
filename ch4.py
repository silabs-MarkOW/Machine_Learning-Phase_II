import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

SHOW_4_10 = True
SHOW_4_11 = True

SAMPLES = 1000
SEED = 1337

np.random.seed(SEED)
tf.random.set_seed(SEED)

x_values = np.random.uniform(low=0, high=2*np.pi, size=SAMPLES)

np.random.shuffle(x_values)

y_values = np.sin(x_values)
y_values += 0.1*np.random.randn(*y_values.shape)

if SHOW_4_10 :
    plt.plot(x_values,y_values,'b.')
    plt.show()

TRAIN_SPLIT = int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES) + TRAIN_SPLIT

split_list = [ TRAIN_SPLIT, TEST_SPLIT ]

x_train, x_validate, x_test = np.split(x_values, split_list)
y_train, y_validate, y_test = np.split(y_values, split_list)

assert  SAMPLES == (x_train.size + x_validate.size + x_test.size)

if SHOW_4_11 :
    plt.plot(x_train, y_train, 'b.', label='Train')
    plt.plot(x_validate, y_validate, 'y.', label='Validate')
    plt.plot(x_test, y_test, 'r.', label='Test')
    plt.legend()
    plt.show()

model_1= tf.keras.Sequential()
model_1.add(layers.Dense(16,activation='relu',input_shape=(1,)))
model_1.add(layers.Dense(1))
model_1.compile(optimzer='rmsprop', loss='mse', metrics=['mae'])
model_1.summary()
