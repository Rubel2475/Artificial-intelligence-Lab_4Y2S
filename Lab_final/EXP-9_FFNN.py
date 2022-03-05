#FFNN- feed forward neural network

from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

m=28
n=28  #image size- m-by-n
h=4   #fully connected hidden layers
c=10  #number of output classes

#input layer
inputs = Input((m, n))
x = Flatten()(inputs)   #convert from 2D into 1D(vector)

#hidden layer
x = Dense(128, activation="softmax")(x)
for i in range(h-1):
  x = Dense(32, activation="softmax")(x)

#output layer
outputs = Dense(c)(x)

#model
model = Model(inputs, outputs)
model.summary()