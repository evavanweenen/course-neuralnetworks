import numpy as np

y_test = np.loadtxt('y_test.txt')
y_train = np.loadtxt('y_train.txt')

print(y_test, y_train)

y_test_yn = [0 if i == 0. else 1 for i in y_test]
y_train_yn = [0 if i == 0. else 1 for i in y_train]

#print(y_test_yn, y_train_yn)

#print(len(np.where(y_test_yn == 'Y')[0]), len(np.where(y_train_yn == 'Y')[0]))
#print(len(np.where(y_test_yn == 'N')[0]), len(np.where(y_train_yn == 'N')[0]))

np.savetxt('y_test_yn.txt', y_test_yn, fmt='%i')
np.savetxt('y_train_yn.txt', y_train_yn, fmt='%i')

