#CNN- Convolutional neural network 

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model

m=28
n=28  #image size- m-by-n
h=4   #convolutional hidden layers
c=10  #number of output classes

#input layer
inputs = Input((m, n, 1))

#hidden layers

# x = Conv2D(filters=3, kernel_size=(2,2), padding="same", strides=(2,2))(inputs)
x = Conv2D(filters=3, kernel_size=(2,2), padding="same")(inputs)    #take either strides or pooling, not together both
x = MaxPooling2D(pool_size=(2,2))(x)    

for i in range(h-1):
  x = Conv2D(filters=3, kernel_size=(2,2), padding="same")(x)  
  x = MaxPooling2D(pool_size=(2,2))(x)

x = Flatten()(x)
#output layer
outputs = Dense(c)(inputs)

model = Model(inputs, outputs)
model.summary()

