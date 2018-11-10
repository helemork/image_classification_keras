from keras.datasets import mnist # standard dataset of hand drawn numbers - digit recognition
import matplotlib.pyplot as plt
import keras
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten
import numpy as np
(x_train, y_train), (x_test,y_test) = mnist.load_data()
x_train = x_train[:10000, :]
y_train = y_train[:10000]
x_test = x_test[:1000, :]
y_test = y_test[:1000]

# x = input (images), y = output (numbers)

print(x_train.shape)
print(y_train.shape)

# Plot some images to see how they look. In grey. With a title.
for i in range(0):
    plt.imshow(x_train[i,:,:], cmap="gray")
    plt.title(y_train[i])
    plt.show()

# have now looked at the data and it looks smashin'
# last 4th dimentions is channels, but we're doing greyscale so dont need that
x_train = np.expand_dims(x_train,axis=-1)
x_test = np.expand_dims(x_test,axis=-1)
print(x_train.shape)

# must now convert images to floats
print(x_train.dtype)
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255
print(x_train.dtype)

# must now convert the output data. It is values, but we want it to match data
# we want onehotencoding
y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
print(y_train.shape)
print(y_test.shape)
print(y_train[0,:])

# MAKING LAYER MAKING NETWORK DUDUDU ITS OURS THIS TIME
first_layer = Input(shape=x_train.shape[1:])
# nr of filters,  size of filter, activationfilter, and x is the prev layer
x = Convolution2D(32, 3, activation="relu",padding="same")(first_layer)
x = Convolution2D(64, 3, activation="relu",padding="same")(x)
# half the width and height
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
# nr of neurons in this layer
x = Dense(128)(x)
x = Dense(128)(x)
x = Dense(10,activation="softmax")(x)

model = keras.Model(inputs=first_layer,outputs=x)
print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer="adadelta")
model.fit(x_train,y_train,batch_size=24,epochs=3,validation_data=(x_test,y_test))
