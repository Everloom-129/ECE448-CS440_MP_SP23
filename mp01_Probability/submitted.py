'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np


def joint_distribution_of_word_counts(texts, word0, word1):
    '''
    Parameters:
    texts (list of lists) - a list of texts; each text is a list of words
    word0 (str) - the first word to count
    word1 (str) - the second word to count

    Output:
    Pjoint (numpy array) - Pjoint[m,n] = P(X1=m,X2=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    '''
    # raise RuntimeError('You need to write this part!')
    x = y = 0
    Pjoint = np.zeros((x+1, y+1))
    # First double loop, used for finding correct size of the Pjoint
    for i_text in texts:
        c0 = c1 = 0
        for i_word in i_text:
            if i_word == word0:
                c0 += 1
            if i_word == word1:
                c1 += 1
            if (x <= c0 or y <= c1):
                # todo: How to use np.pad()?
                # d1 = max(0, c0 - x)
                # d2 = max(0, c1 - y)
                # Pjoint = np.pad(Pjoint,((0,d2),(0,d1)), 'constant')
                x, y = max(x, c0), max(y, c1)
    Pjoint = np.zeros((x+1, y+1))
    # Second double loop, used for filling the Pjoint
    for i_text in texts:
        c0 = c1 = 0
        for i_word in i_text:
            if i_word == word0:
                c0 += 1
            if i_word == word1:
                c1 += 1
        Pjoint[c0][c1] += 1
    return Pjoint/len(texts)


def marginal_distribution_of_word_counts(Pjoint, index):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word1 occurs in a given text,
      X1 is the number of times that word2 occurs in the same text.
    index (0 or 1) - which variable to retain (marginalize the other) 

    Output:
    Pmarginal (numpy array) - Pmarginal[x] = P(X=x), where
      if index==0, then X is X0
      if index==1, then X is X1
      Test Failed: index 7 is out of bounds for axis 1 with size 7
    '''
    # First, get the shape of Pjoint 
    x = np.size(Pjoint, 0)
    y = np.size(Pjoint, 1)
    if index == 0:
        Pmarginal = np.zeros(x)
        for i in range(x):
            Pmarginal[i] = np.sum(Pjoint[i, :])
    else:
        Pmarginal = np.zeros(y)
        for i in range(y):
            Pmarginal[i] = np.sum(Pjoint[:, i])

    return Pmarginal


def conditional_distribution_of_word_counts(Pjoint, Pmarginal):
    '''
    Parameters:
    Pjoint (numpy array) - Pjoint[m,n] = P(X0=m,X1=n), where
      X0 is the number of times that word0 occurs in a given text,
      X1 is the number of times that word1 occurs in the same text.
    Pmarginal (numpy array) - Pmarginal[m] = P(X0=m)

    Outputs: 
    Pcond (numpy array) - Pcond[m,n] = P(X1=n|X0=m)
    '''
    Pcond = np.zeros(Pjoint.shape)
    x = np.size(Pjoint, 0)
    y = np.size(Pjoint, 1)
    for i in range(x):
        for j in range(y):
            if Pmarginal[i] != 0:
                Pcond[i][j] = Pjoint[i][j] / Pmarginal[i]
            else:
                Pcond[i][j] = np.nan
    return Pcond


def mean_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    mu (float) - the mean of X
    '''
    mu = 0.0
    length = P.size
    for i in range(length):
        mu += (i)*P[i]
    return mu


def variance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[n] = P(X=n)

    Outputs:
    var (float) - the variance of X
    '''
    var = 0.0
    mu = mean_from_distribution(P)
    for i in range(P.size):
        var += pow((i-mu),2) * P[i]
    return var


def covariance_from_distribution(P):
    '''
    Parameters:
    P (numpy array) - P[m,n] = P(X0=m,X1=n)

    Outputs:
    covar (float) - the covariance of X0 and X1
    cov[x,y] = E[xy] - E[x]E[y]

    '''
    covar = 0.0
    P_x = marginal_distribution_of_word_counts(P,0)
    P_y = marginal_distribution_of_word_counts(P,1)

    mu_x = mean_from_distribution(P_x)
    mu_y = mean_from_distribution(P_y)
    # print(mu_x,mu_y)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            covar += P[i][j] * i * j
    covar -= mu_x * mu_y



    return covar 


def expectation_of_a_function(P, f):
    '''
    Parameters:
    P (numpy array) - joint distribution, P[m,n] = P(X0=m,X1=n)
    f (function) - f should be a function that takes two
       real-valued inputs, x0 and x1.  The output, z=f(x0,x1),
       must be a real number for all values of (x0,x1)
       such that P(X0=x0,X1=x1) is nonzero.

    Output:
    expected (float) - the expected value, E[f(X0,X1)]
    '''
    expected = 0.0
    x = P.shape[0]
    y = P.shape[1]
    for i in range(x):
        for j in range(y):
            expected += f(i,j) *P[i][j]
    return expected
