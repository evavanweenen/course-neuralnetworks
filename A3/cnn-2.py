'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
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
num_classes = 2
epochs = 4 

# input image dimensions
img_rows, img_cols = 100, 100

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
x_train = np.load(root+'x_train.npy')#[:4004]
x_test = np.load(root+'x_test.npy')#[:2002] #alleen merger data

y_train_in =  np.loadtxt(root+'y_train_yn.txt', dtype='int')[:len(x_train)]#[:4004]
y_test_in = np.loadtxt(root+'y_test_yn.txt', dtype='int')[:len(x_test)]#[:2002]

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

print("x_train[0]", x_train[0])
print("y_train[0]", y_train[0])

print(input_shape)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5),
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
np.savetxt('predict-binary-complete.txt', predict, fmt='%i')
predict_wrong = []
for i in range(len(predict)):
	if predict[i] != y_test_in[i] and predict[i] != y_test_in[i] + 1. and predict[i] != y_test_in[i] - 1.:
		predict_wrong.append(i)
predict_wrong = np.array(predict_wrong)
np.savetxt('predict_wrong-binary-complete.txt', predict_wrong, fmt='%i')
#indices of images that are misclassified (can still be neighbouring class)
#probability = np.amax(p, axis=-1)
#probability_wrong = probability[predict_wrong] #prediction probability misclassified images

misclassification = np.zeros((num_classes, num_classes))

for i in range(len(predict)):
	misclassification[predict[i],y_test_in[i]] += 1

plt.imshow(misclassification, cmap='gray')
plt.savefig('confusion-bin.png')
"""
print(predict_wrong[0][probability_wrong > 0.997])
print(predict_wrong[0][np.argsort(probability_wrong)[-3:]])

for i in predict_wrong[0][np.argsort(probability_wrong)[-3:]]:
    plt.title("True value %s ; prediction %s ; probability %s"%(y_test2[i], predict[i], probability[i]))
    plt.imshow(x_test[i].reshape(28,28))
    plt.savefig(root+'mostmisclassified_mlp_shuffle_mse_try%s_image%s.pdf'%(number,i))

"""

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
	#if predict[i] == y_test_in[i] or predict[i] == y_test_in[i]+1. or predict[i] == y_test_in[i]-1.: # classified in the right class, or 1 class off
		#accuracy.append(1.)
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

 
for i in wrong_indices:
	label = y_test_in[i]
	predicted = predict[i]
	if i%10 == 1:
		print(i)
		print('WRONG. Label: ',label, 'Predicted: ', predicted)
		plt.imshow(x_test[i].reshape(100,100))
		plt.savefig('Images/Wrong_in%s_out%s_nr%s.png'%(label, predicted, i))
		#plt.show()

for i in right_indices:
        label = y_test_in[i]
        predicted = predict[i]
        if i%10 == 1:
                print(i)
                print('RIGHT. Label: ',label, 'Predicted: ', predicted)
                plt.imshow(x_test[i].reshape(100,100))
                plt.savefig('Images/Right_in%s_out%s_nr%s.png'%(label, predicted, i))
#i = wrong_indices[50]
#print((x_test[i].reshape(100,100)).shape)
#plt.imshow(x_test[i].reshape(100,100)) 
#plt.show()

