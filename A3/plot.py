import numpy as np
from matplotlib import pyplot as plt

y_test = np.loadtxt('y_test_yn.txt')[:5255]
predict = np.loadtxt('predict-binary-complete.txt')
y_test = y_test.astype(int)
predict = predict.astype(int)

print(len(y_test))
print(len(predict))
c = 2

def Number(data):
    """
    Input:
        data - vector of all output labels
    Returns:
        array where ith element contains number of pictures for digit i
    """
    #number of pictures for each digit
    return [len(data[np.where(data == i)]) for i in range(11)]


def ConfusionMatrix(fit, data_out, n):
    """
    Calculate confusion matrix where i,jth element contains percentage of digits i classified as j
    Input:
        data_in - all images
        fit - predicted value for each image (vector)
        data_out - desired value for each image (vector)
        n - number of pictures for each digit (10-vector)
    Returns:
        confusion - 10x10 confusion matrix
    """
    confusion = np.zeros((c,c))
    for i in range(c):
        for j in range(c):
            confusion[j,i] = len(np.where((data_out == i) & (fit == j))[0])/n[i]*1000.
    return confusion

confusion = ConfusionMatrix(predict, y_test, Number(y_test))
plt.figure(1)
plt.imshow(confusion, cmap='gray')
plt.savefig('confusion-binary-complete.png')

distr = np.zeros((c))
for i in y_test:
	distr[int(i)] += 1.
#plt.imshow(distr.reshape(1,11), cmap='gray')
#plt.savefig('distr-complete.png')
distr = np.array(distr)
print(y_test)
print(np.where(y_test == 2))
plt.figure(2)
plt.hist(y_test)
plt.savefig('distr-hist.png')


