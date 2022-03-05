#train_test_CNN => train and test a convolutional neural network for binary classification.

import numpy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Model


(trainX, trainY), (testX, testY) = mnist.load_data()
train_indices = numpy.argwhere((trainY==1) | (trainY==2))   #index(singular) -> indices(plural)
test_indices = numpy.argwhere((testY == 1) | (testY==2))    #two number selected for binary classification
train_indices = numpy.squeeze(train_indices)
test_indices = numpy.squeeze(test_indices)

trainX = trainX[train_indices]
trainY = trainY[train_indices]
testX = testX[test_indices]
testY = testY[test_indices]

trainY = to_categorical(trainY==1, num_classes=2)
testY = to_categorical(testY==1, num_classes=2)

trainX = trainX.astype(numpy.float32)
testX = testX.astype(numpy.float32)

trainX /= 255
testX /= 255

m = trainX.shape[1]
n = trainX.shape[2]
h = 3

inputs = Input((m, n, 1))

x = Conv2D(filters=3, kernel_size=(2,2), padding="same")(inputs)
x = MaxPooling2D(pool_size=(2,2))(x)

for i in range(h-1):
  x = Conv2D(filters=3, kernel_size=(2,2), padding="same")(x)
  x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
outputs = Dense(2)(x)

model = Model(inputs, outputs)

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
model.fit(trainX, trainY, epochs=10, validation_split=0.2)
model.evaluate(testX, testY)





