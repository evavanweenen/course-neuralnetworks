import numpy as np

root = '/disks/strw9/willebrandsvanweenen/assignment3/data/'

merger = np.genfromtxt(root + 'darg_mergers.csv', skip_header=1, dtype='float', delimiter=',', usecols=(50,51)) 
#merger galaxies
#array aanmaken voor y_train: merg + nomerg
nomerg = np.zeros((10000))

y_train_merg1 = merger[:2002,0]
y_train_merg2 = merger[:2002,1]
y_train_nomerg = np.zeros((6700))

y_test_merg1 = merger[2002:,0]
y_test_merg2 = merger[2002:,1]
y_test_nomerg = np.zeros((3300))


#if merg1 and merg2 are the same object
for i in range(len(y_train_merg2)):
    if y_train_merg2[i] > 1. or y_train_merg2[i] < 0.:
        y_train_merg2[i] = y_train_merg1[i]
        
for i in range(len(y_test_merg2)):
    if y_test_merg2[i] > 1. or y_test_merg2[i] < 0.:
        y_test_merg2[i] = y_test_merg1[i]


y_train = np.concatenate((y_train_merg1, y_train_merg2, y_train_nomerg))
print(y_train.shape)
y_test = np.concatenate((y_test_merg1, y_test_merg2, y_test_nomerg))
print(y_test.shape)

y_train = np.int_(np.around(y_train*10.,0))
y_test = np.int_(np.around(y_test*10.,0))

print(y_train,y_test)

np.savetxt(root+'y_train.txt', y_train, fmt='%i')
np.savetxt(root+'y_test.txt', y_test, fmt='%i')
