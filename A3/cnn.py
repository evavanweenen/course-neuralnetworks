'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import time
#from keras.preprocessing.image import save_img
from keras.applications import vgg16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os

root = '/home/s1338161/nn3/'

batch_size = 128 
num_classes = 11
epochs = 8

# input image dimensions
img_rows, img_cols = 100, 100

layer_name = 'conv_layer2'

def OpenImage(names, dirs):
    x_arr = []
    for name in names:
        im = Image.open(root+dirs+name)
        x = np.array(im.getdata())
        x = np.array(x)
        x = np.average(x.reshape(100,100,3), axis=2)
        x_arr.append(x)
        im.close()
    print("Done opening images for ..", dirs)
    return np.array(x_arr)
"""
def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + K.epsilon())
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	if K.image_data_format() == 'channels_first':
		x = x.transpose((1, 2, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x
"""
"""
#edit filenames
names = os.listdir(root+'/test/merg2/')
newnames = []

for name in names:
    newname, rest = name.split('ra')
    newname = newname.strip('im')
    newnames.append(newname)

print(newnames)

for i in range(len(names)):
    #print(root+'/train/merg2/'+names[i])
    os.rename(root+'/test/merg2/'+names[i], root+'/test/merg2/'+newnames[i])

# import cropped data
names_train_merg1 = sorted(os.listdir(root+'/train/merg1/'), key=int)
names_train_merg2 = sorted(os.listdir(root+'/train/merg2/'), key=int)
names_train_nomerg = sorted(os.listdir(root+'/train/nomerg/'), key=int)
names_test_merg1 = sorted(os.listdir(root+'/test/merg1/'), key=int)
names_test_merg2 = sorted(os.listdir(root+'/test/merg2/'), key=int)
names_test_nomerg = sorted(os.listdir(root+'/test/nomerg/'), key=int)

x_train_merg1 = OpenImage(names_train_merg1, '/train/merg1/')
x_train_merg2 = OpenImage(names_train_merg2, '/train/merg2/')
x_train_nomerg = OpenImage(names_train_nomerg, '/train/nomerg/')
x_test_merg1 = OpenImage(names_test_merg1, '/test/merg1/')
x_test_merg2 = OpenImage(names_test_merg2, '/test/merg2/')
x_test_nomerg = OpenImage(names_test_nomerg, '/test/nomerg/')

x_train = np.concatenate((x_train_merg1, x_train_merg2, x_train_nomerg))
x_test = np.concatenate((x_test_merg1, x_test_merg2, x_test_nomerg))

np.save(root+'x_train.npy', x_train)
np.save(root+'x_test.npy', x_test)
"""
x_train = np.load(root+'x_train.npy')[:4004]
x_test = np.load(root+'x_test.npy')[:2002] #alleen merger data

y_train_in =  np.loadtxt(root+'y_train.txt', dtype='int')[:4004]
y_test_in = np.loadtxt(root+'y_test.txt', dtype='int')[:2002]

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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train_in, num_classes)
y_test = keras.utils.to_categorical(y_test_in, num_classes)

print(input_shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
hist.history['loss']
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(hist)
acc_arr = hist.history['acc']
loss_arr = hist.history['loss']
valacc_arr = hist.history['val_acc']
valloss_arr = hist.history['val_loss']
history = np.vstack((acc_arr, loss_arr, valacc_arr, valloss_arr))
print(history)
np.save('hist_arr.npy', history)

p = model.predict(x_test, batch_size=batch_size, verbose=1) # vector of probabilities per class
predict = np.argmax(p, axis=-1) #predicted digits of test data
predict_wrong = np.nonzero(predict != y_test_in) #indices of images that are misclassified (can still be neighbouring class)
probability = np.amax(p, axis=-1)
probability_wrong = probability[predict != y_test_in] #prediction probability misclassiefied images
#probability_more_wrong = probability[np.where((predict != y_test_in) & (predict != y_test_in+1.) & (predict != y_test_in-1.))]

print('HIER KIJKEN', predict)
print(predict[:200])
# calculate accuracy: prediction is ok when it's one class off
accuracy = []
accuracy_check = []
wrong_indices = []
right_indices = []

for i in range(len(predict)):
	if predict[i] == y_test_in[i]: # classified in exactly he right class
		accuracy_check.append(1.)
		right_indices.append(i)
	if predict[i] == y_test_in[i] or predict[i] == y_test_in[i]+1. or predict[i] == y_test_in[i]-1.: # classified in the right class, or 1 class off
		accuracy.append(1.)
	else:
		wrong_indices.append(i)

wrong_indices = np.array(wrong_indices)
right_indices = np.array(right_indices)
acc = len(accuracy)/len(predict) * 100.
acc_check = len(accuracy_check)/len(predict) * 100. # moet gelijk zijn aan val_acc
print('plus-min accuracy: ', acc)
print('categorical accuracy: ', acc_check)

#print('Predict wrong: ', predict_wrong)
#print('Predict more wrong: ', wrong_indices)
#print(predict.shape, predict_wrong[0].shape, wrong_indices.shape)

''' 
for i in wrong_indices:
	label = y_test_in[i]
	predicted = predict[i]
	#if i%100 == 1:
	if label - predicted > 4:	
		print(i)
		print('WRONG. Label: ', label, ' Predicted: ', predicted)
		plt.imshow(x_test[i].reshape(100,100))
		plt.show()

for i in right_indices:
	label = y_test_in[i]
	predicted = predict[i]
	if i%50 == 1:
		print(i)
		print('RIGHT. Label: ',label, 'Predicted: ', predicted)
		plt.imshow(x_test[i].reshape(100,100))
		plt.show()
'''		
#i = wrong_indices[50]
#print((x_test[i].reshape(100,100)).shape)
#plt.imshow(x_test[i].reshape(100,100)) 
#plt.show()















