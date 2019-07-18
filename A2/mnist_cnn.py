'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

root = '/disks/strw9/willebrandsvanweenen/assignment2/'

batch_size = 128
num_classes = 10
epochs = 1 #12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test2) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#x_train = (np.random.permutation(x_train.reshape(60000,28*28).T).T).reshape(60000,28,28,1)
#x_test = (np.random.permutation(x_test.reshape(10000,28*28).T).T).reshape(10000,28,28,1)

#shuffle array
x_shuffle = np.random.permutation(np.concatenate((x_train.reshape(60000,28*28).T, x_test.reshape(10000,28*28).T), axis=1))
x_train = x_shuffle[:,:60000].T.reshape(60000,28,28,1)
x_test = x_shuffle[:,60000:].T.reshape(10000,28,28,1)
print(x_train.shape)
print(x_test.shape)

plt.imshow(x_test[0].reshape(28,28))
plt.show()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test2, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

#model.compile(loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
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

print(predict_wrong[0][np.argsort(probability_wrong)[-3:]])

for i in predict_wrong[0][np.argsort(probability_wrong)[-3:]]:
    plt.title("True value %s ; prediction %s ; probability %s"%(y_test2[i], predict[i], probability[i]))
    plt.imshow(x_test[i].reshape(28,28))
    #plt.savefig(root+'mostmisclassified_cnn_%s.pdf'%i)
    plt.show()
