import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances as pairdist
import scipy.spatial.distance
from matplotlib import rc
rc('text', usetex=True)

root = '/disks/strw9/willebrandsvanweenen/assignment1/'

test_in = np.loadtxt(root + '/data/test_in.csv', delimiter=',')
test_out = np.loadtxt(root + '/data/test_out.csv', delimiter=',', dtype=int)
train_in = np.loadtxt(root + '/data/train_in.csv', delimiter=',')
train_out = np.loadtxt(root + '/data/train_out.csv', delimiter=',', dtype=int)

#Task 1: Calculate center, radius and number of images for every digit and calculate distance between centers

def EuclidianDistance(p,q):
    """
    Calculate the Euclidian distance between two vectors p and q
    Input:
        p - i dimensional vector
        q - i dimensional vector
    Returns:
        np.sqrt(d2) - Euclidian distance (scalar)
    """
    d2 = 0
    for i in range(len(p)):
        d2 += (p[i] - q[i])**2.
    return np.sqrt(d2)

def Number(data):
    """
    Input:
        data - vector of all output labels
    Returns:
        array where ith element contains number of pictures for digit i
    """
    #number of pictures for each digit
    return [len(data[np.where(data == i)]) for i in range(10)]

def Center(data_in, data_out):
    """
    Calculate center for each digit cloud (calculate average image for each digit)
    Input:
        data_in - all images
        data_out - the labels of all images
    Returns:
        center (256x10-matrix) where each ith column contains the average 256-vector for digit i
    """
    return [np.mean(data_in[np.where(data_out == i)], axis=0) for i in range(10)]

def Radius(data_in, data_out, n, c):
    """
    Calculate the maximum distance of points from the center for every cloud
    Input:
        data_in - all images
        data_out - the labels of all images
        n - number of pictures for each digit
        c - centers of each digit (256x10-matrix) 
    Returns:
        r - radius (maximum distance from center) for each digit (10-vector) 
    """
    r = np.empty((10))
    for i in range(10):
        for j in range(n[i]):
            temp = EuclidianDistance(c[i], data_in[np.where(data_out == i)][j])
        if temp > r[i]:
            r[i] = temp
    return r

def CenterDistance(c):
    """
    Calculate distance between all centers
    Input:
        c - centers of each digit (256x10-matrix)
    Returns:
        dc - distances between all centers (10x10-matrix)
    """    
    dc = np.empty((10,10))
    for i in range(10):
        for j in range(10):
            dc[i,j] = EuclidianDistance(c[i], c[j])
    return dc

number_train = Number(train_out)
center = Center(train_in, train_out)
radius = Radius(train_in, train_out, number_train, center)
radiusnorm = radius / number_train
centerdistance = CenterDistance(center)

print("Number of images per digit\n", number_train)
print("Radius per digit\n", radius)
print("Normalised radius\n", radiusnorm)
print("Distance between centers\n", np.around(centerdistance, decimals=1))

# Plot centers for every cloud
fig, ax = plt.subplots(2,5, figsize=(12,4), tight_layout=True)
for i, cent in enumerate(center):
	ax[i//5,i%5].imshow(cent.reshape(16,16), cmap='binary')
	ax[i//5,i%5].xaxis.set_major_formatter(plt.NullFormatter())
	ax[i//5,i%5].yaxis.set_major_formatter(plt.NullFormatter())
plt.savefig("centers.png")

plt.matshow(centerdistance, cmap='binary')
plt.suptitle('Center distance')
plt.savefig("centerdistance.png")


# Task 2: Apply distance classifier, calculate accuracy, confusion matrix and wrongly classified digits

def ApplyClassifierEuclidian(data_in, c):
    """
    Apply distance-based classifier (only Euclidian)
    Input:
        data_in - all images
        c - centers of digits from training set
    Returns:
        fit - predicted value for each image (vector)
    """
    diff = np.empty((len(data_in),10))
    for n in range(len(data_in)): #over all pictures 
        for i in range(10): #over all digits
            diff[n,i] = EuclidianDistance(c[i], data_in[n])
    fit = np.argmin(diff, axis=1)
    return fit

def ApplyClassifierPairwise(data_in, c, metric):
    """
    Apply distance-based classifier (use pairwise distance)
    Input:
        data_in - all images
        c - centers of digits from training set
        metric - metric used to calculate distance in
    Returns:
        fit - predicted value for each image (vector)
    """
    diff = np.empty((len(data_in),10))
    for n in range(len(data_in)): #over all pictures 
        for i in range(10): #over all digits
            diff[n,i] = pairdist(c[i].reshape(1,-1), data_in[n].reshape(1,-1), metric)
    fit = np.argmin(diff, axis=1)
    return fit

def Accuracy(fit, data_out):
    """
    Calculate percentage of correctly classified digits
    Input:
        fit - predicted value for each image (vector)
        data_out - desired value for each image (vector)
    Returns:
        percentage of correctly classified digits
    """
    return len(np.where(fit == data_out)[0])/len(data_out)*100.

def ConfusionMatrix(data_in, fit, data_out, n):
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
    confusion = np.empty((10,10))
    for i in range(10):
        for j in range(10):
            confusion[i,j] = len(np.where((fit != data_out) & (data_out == i) & (fit == j))[0])/n[i]*100.
    return confusion

def ArgWrong(confusion, percentage = 5.):
    """
    Input:
        confusion - confusion matrix containing percentage of misclassified examples
        percentage - boundary percentage
    Returns
        indices of confusion matrix where the percentage of misclassified examples is above 'percentage'
    """
    return np.argwhere(confusion > percentage)

#Apply distance-based classifier to all points of training set
fit_train = ApplyClassifierEuclidian(train_in, center)
accuracy_train = Accuracy(fit_train, train_out)
confusion_train = ConfusionMatrix(train_in, fit_train, train_out, number_train) 
wrong_train = ArgWrong(confusion_train)

plt.matshow(confusion_train, cmap='binary')
plt.suptitle('Confusion matrix training data')
plt.savefig("confusionmatrix-training.png")

print("Accuracy training data", np.around(accuracy_train, decimals=2))
print("Wrongly classified digits above 5 percent\n", wrong_train)
print("Confusion matrix training data\n", np.around(confusion_train, decimals=1))

#Apply distance-based classifier to all points of test set
number_test = Number(test_out)
fit_test = ApplyClassifierEuclidian(test_in, center)
accuracy_test = Accuracy(fit_test, test_out)
confusion_test = ConfusionMatrix(test_in, fit_test, test_out, number_test) 
wrong_test = ArgWrong(confusion_test)

plt.matshow(confusion_test, cmap='binary')
plt.suptitle('Confusion matrix test data')
plt.savefig("confusionmatrix-test.png")

print("Accuracy test data", np.around(accuracy_test, decimals=2))
print("Wrongly classified digits above 5 percent\n", wrong_test)
print("Confusion matrix test data\n", np.around(confusion_test, decimals=1))

#Apply distance-based classfier to all points of test set using different metrics
metrics = ('cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule')
accuracy_metrics = np.empty((len(metrics)))
for i in range(len(metrics)):
    fit_i = ApplyClassifierPairwise(np.array(test_in), np.array(center), metrics[i])
    accuracy_metrics[i] = Accuracy(fit_i, test_out)
    confusion_i = ConfusionMatrix(test_in, fit_i, test_out, number_test) 
print("Accuracy for each metric:\n", metrics, accuracy_metrics.T)
print("Best accuracy metric:", metrics[np.argmax(accuracy_metrics)])


# Task 3: Implement a Bayes Rule classifier

def feature_width(images):
    """
    Calculate width of image
    Input:
        images - images in a class
    Returns:
        width - width of images (vector)
    """
    proj = np.zeros((len(images), 16))
    width = np.zeros((len(images)))
    for i, im in enumerate(images):
        im = (im *(im>=0.)).reshape(16,16)
        proj[i,:] = np.sum(im, axis=0)
        width[i] = len(np.trim_zeros(proj[i,:]))
    return width

#Images in categories
C0 = train_in[np.where(train_out == 0)]
C1 = train_in[np.where(train_out == 1)]
print(len(C0), len(C1))

#Apply feature width to categories 0 and 1
width_C0 = feature_width(C0)
width_C1 = feature_width(C1)

#Plot histogram of feature width
plt.figure(10)
plt.title('Histogram of feature')
plt.hist(width_C0, label='C0', range=(0.5,16.5), bins=8, alpha=.5)
plt.hist(width_C1, label='C1', range=(0.5,16.5), bins=8, alpha=.5)
plt.xlabel('width') ; plt.ylabel('$P(C_i | x)$')
plt.legend()
plt.savefig("histbayes.png")

#Calculate probabilities
hist0, edges0 = np.histogram(width_C0, range=(0.5, 16.5), bins=8)
hist1, edges1 = np.histogram(width_C1, range=(0.5, 16.5), bins=8)

P_C0_x = hist0/(hist0+hist1)
P_C1_x = hist1/(hist0+hist1)

P_C0_x_plot = np.append(P_C0_x, P_C0_x[-1])
P_C1_x_plot = np.append(P_C1_x, P_C1_x[-1])

#Set boundary of feature
boundary = 6.5

#Plot conditional probability
plt.figure(11)
plt.title('Stepfunction of feature')
plt.step(edges0, P_C0_x_plot, label='C0', where='post')
plt.step(edges1, P_C1_x_plot, label='C1', where='post')
plt.axvline(boundary, linestyle='dashed', color='red', label='Decision boundary')
plt.xlabel('width') ; plt.ylabel('$P(C_i | x)$')
plt.legend()
plt.savefig("condprob-bayes.png")

#Determine accuracy of training data
accuracy_Bayes_training = (1. - (np.sum(hist0[:3]) + np.sum(hist1[3:]))/(len(C0)+len(C1)))*100.
print("Accuracy Bayes classifier training data: ", np.round(accuracy_Bayes_training, 1), '%')

#Determine accuracy of test data
test_C01 = test_in[np.where((test_out == 0) | (test_out == 1))]
test_C01_out = test_out[np.where((test_out == 0) | (test_out == 1))]

width_test_C01 = feature_width(test_C01)
test_C01_fit = np.array([1 if (w <= 6.5) else 0 for w in width_test_C01])
accuracy_Bayes_test = Accuracy(test_C01_fit, test_C01_out)
print("Accuracy Bayes classifier test data: ", np.round(accuracy_Bayes_test, 1), '%')


# Task 4: Implement Multi-Class (Single-Layer) Perceptron algorithm

def edit_images(ims):
    """
    Insert scalar 1 into every image which is later multiplied by bias w0
    """
    return np.insert(ims, 0, 1., axis=1)

def predict(w, ims):
	"""
	Calculate discriminant 
	    z = w0 + w1x1 + ... + w256x256 for 10 nodes
	Calculate prediction
	    fit = argmax(z)
	Input:
	    w - weights
	    ims - images
	Returns:
	    fit - prediction (binary 10 vector for every image)
	"""
	fit = np.zeros((len(ims), 10))
	for i, im in enumerate(ims):
		z = np.dot(w, im) #discriminant
		fit[i, np.argmax(z)] = 1 #prediction
	return fit

def desired(out):
    """
    Calculate desired output in binary 10 vector. Binary 10 vector is 1 at element 'out' and 0 elsewhere.
    Input:
        out - desired output (scalar for every image)
    Returns:
        des - desired output (binary 10 vector for every image)
    """
    des = np.zeros((len(out), (10)))
    for i, nr in enumerate(out):
        des[i, nr] = 1
    return des

def perceptron(data_in, data_out):
    """
    Apply perceptron algorithm
    Input:
        data_in
        data_out
    Calculates:
        eta - stepwidth
        w - set of random initialized weights for every pixel and for every node
        des - desired output (binary 10 vector for every image) (shape: len(data_out), 10)
        fit - predicted output (binary 10 vector for every image)
        y - label of predicted output (scalar for every image) (same shape as data_out)
        counter - keeps track of epoch
        acc - accuracy for every epoch
        r - index of a random misclassified sample
    Returns:
        y - best y
        w - best set of weights
    """
    eta = 1.
    w = np.random.rand(10,257) 
    des = desired(data_out) #desired output (binary)
    fit = predict(w, data_in) #predicted output (binary)
    y = np.argmax(fit, axis=1) #desired output (scalar)
    epoch = 0
    acc = []
    while not np.array_equal(data_out, y):
        r = np.random.choice(np.nonzero(y != data_out)[0])
        w += eta*np.dot((des[r]-fit[r]).reshape(10,1), data_in[r].reshape(1,257)) #update weights
        acc.append(Accuracy(y, data_out))
        fit = predict(w, data_in)
        y = np.argmax(fit, axis=1)
        epoch += 1.
        if epoch%100 == 0:
            print(epoch)
            print(Accuracy(y, data_out))
    print("100% Accuracy reached aka network training completed")
    return w, acc
    
train_in = edit_images(train_in)
test_in = edit_images(test_in)

weights_perc, accuracy_perc = perceptron(train_in, train_out)

print("Overfitted weights\n", len(np.where(np.ravel(weights_perc) > 10.)[0]))

fit_perc_test = np.argmax(predict(weights_perc, test_in), axis=1)

accuracy_perc_test = Accuracy(fit_perc_test, test_out)
print("Accuracy Perceptron Algorithm on test data: ", np.round(accuracy_perc_test, 1), '%')

plt.figure()
plt.title("Evolution of accuracy and mse during training of network")
plt.plot(np.arange(len(accuracy_perc)), accuracy_perc, linewidth=.3, color='green')
plt.xlabel("epoch") ; plt.ylabel("Accuracy perceptron")
fig.savefig("training-accuracy.png")
plt.show()
