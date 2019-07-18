'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

root = '/disks/strw9/willebrandsvanweenen/assignment2/'

batch_size = 128
num_classes = 10
epochs = 20 #20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test2) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#shuffle array
x_shuffle = np.random.permutation(np.concatenate((x_train.T, x_test.T), axis=1))
x_train = x_shuffle[:,:60000].T
x_test = x_shuffle[:,60000:].T
print(x_train.shape)
print(x_test.shape)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test2, num_classes)

model = Sequential() #maak netwerk met lagen
model.add(Dense(512, activation='relu', input_shape=(784,))) #add dense layer??
model.add(Dropout(0.2)) #20% van connecties disablen om overfitten te voorkomen
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

#model.compile(loss='mean_squared_error', optimizer=RMSprop(), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',#error functie 
              optimizer=RMSprop(), #bijv. gradient descent
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

p = model.predict(x_test, batch_size=batch_size, verbose=1)
predict = np.argmax(p, axis=-1) #predicted digits fo all test data
predict_wrong = np.nonzero(predict != y_test2) #indices of images that are misclassified
probability = np.amax(p, axis=-1)
probability_wrong = probability[predict != y_test2] #prediction probability misclassiefied images

print(predict_wrong[0][probability_wrong > 0.997])
print(predict_wrong[0][np.argsort(probability_wrong)[-3:]])

for i in predict_wrong[0][np.argsort(probability_wrong)[-3:]]:
    plt.title("True value %s ; prediction %s ; probability %s"%(y_test2[i], predict[i], probability[i]))
    plt.imshow(x_test[i].reshape(28,28))
    #plt.savefig(root+'mostmisclassified_mlp_%s.pdf'%i)
    plt.show()
