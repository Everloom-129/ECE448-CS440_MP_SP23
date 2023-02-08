'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np
import warnings # Cite idea from https://blog.csdn.net/xc_zhou/article/details/88130541

def k_nearest_neighbors(image, train_images, train_labels, k):
    '''
    Parameters:
    image - one image
    train_images - a list of N images
    train_labels - a list of N labels corresponding to the N images
    k - the number of neighbors to return

    Output:
    neighbors - a list of k images, the k nearest neighbors of image
    labels - a list of k labels corresponding to the k images
    '''
    neighbors = []
    distance = []
    labels = []
    index_array = []
    for picture in train_images:
        euclidean_distance = np.linalg.norm(np.array(picture) - np.array(image))
        distance.append(euclidean_distance)
    index_array = np.argsort(distance)
    neighbors = train_images[index_array[list(range(k))]]
    labels = train_labels[index_array[list(range(k))]]
    return neighbors, labels


def classify_devset(dev_images, train_images, train_labels, k):
    '''
    Parameters:
    dev_images (list) -M images
    train_images (list) -N images
    train_labels (list) -N labels corresponding to the N images
    k (int) - the number of neighbors to use for each dev image

    Output:
    hypotheses (list) -one majority-vote labels for each of the M dev images
    scores (list) -number of nearest neighbors that voted for the majority class of each dev image
    '''
    hypotheses = []
    scores = []
    # One loop
    for image in dev_images:
        neighbors, labels = k_nearest_neighbors(image, train_images, train_labels, k)
        if (sum(labels)> k/2):
            hypotheses.append(True)
            scores.append(sum(labels))
        else:
            hypotheses.append(False)
            scores.append(k - sum(labels))
    return hypotheses, scores


def confusion_matrix(hypotheses, references):
    '''
    Parameters:
    hypotheses (list) - a list of M labels output by the classifier
    references (list) - a list of the M correct labels

    Output:
    confusions (list of lists, or 2d array) - confusions[m][n] is 
    the number of times reference class m was classified as
    hypothesis class n.
    accuracy (float) - the computed accuracy
    f1(float) - the computed f1 score from the matrix
    '''
    accuracy = 0.0
    f1 = 0.0
    # no loop 
    TN = FP = FN = TP = 0
    N = len(hypotheses)
    for i,j in zip(hypotheses,references):
        TN += (i==j and i == 0)
        FN += (i!=j and i == 0)
        FP += (i!=j and i == 1)
        TP += (i==j and i == 1)
    confusions = np.array( [[TN,FP],[FN,TP]])
    
    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    accuracy = (TP+TN) / N
    f1 = 2/(1/Recall+1/Precision)
    return confusions, accuracy,f1
