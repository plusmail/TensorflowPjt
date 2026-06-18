import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD

x_data = np.array([1,2,3,4,5,6])
t_data = np.array([3,4,5,6,7,8])

model = Sequential()
model.add(Flatten(input_shape=(1,))) #입력
model.add(Dense(1, activation='linear')) #출력

model.compile(optimizer=SGD(learning_rate=1e-2), loss='mse')

model.summary()


# W = tf.Variable(tf.random.normal([1]))
# print("초기값 W=" , W.numpy())
#
# for step in range(2):
#     W = W + 1.0
#     print('step=', step, ',W=', W.numpy())
