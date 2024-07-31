'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
###################################
        # ECE 448 MP02
        # Jie Wang
        # jiew5
        # 1/30/2023
###################################

# Note: discussed some idea with my dear roommate Yitao Cai [netid: yitaoc3]

import numpy as np
from collections import Counter

stopwords = set(["a","about","above","after","again","against","all","am","an","and","any","are","aren","'t","as","at","be","because","been","before","being","below","between","both","but","by","can","cannot","could","couldn","did","didn","do","does","doesn","doing","don","down","during","each","few","for","from","further","had","hadn","has","hasn","have","haven","having","he","he","'d","he","'ll","he","'s","her","here","here","hers","herself","him","himself","his","how","how","i","'m","'ve","if","in","into","is","isn","it","its","itself","let","'s","me","more","most","mustn","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan","she","she","'d","she","ll","she","should","shouldn","so","some","such","than","that","that","the","their","theirs","them","themselves","then","there","there","these","they","they","they","they","'re","they","this","those","through","to","too","under","until","up","very","was","wasn","we","we","we","we","we","'ve","were","weren","what","what","when","when","where","where","which","while","who","who","whom","why","why","with","won","would","wouldn","you","your","yours","yourself","yourselves"])

def create_frequency_table(train):
    '''
    Parameters:
    train (dict of list of lists) 
        - train[y][i][k] = k'th token of i'th text of class y

    Output:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y
    '''
    frequency = {}
    for class_type in train.keys():
        frequency[class_type] = Counter()
        for text in train[class_type]:  
            for word in text:
                    frequency[class_type][word] += 1

    return frequency

def remove_stopwords(frequency):
    '''
    Parameters:
    frequency (dict of Counters) 
        - frequency[y][x] = number of tokens of word x in texts of class y

    Output:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in texts of class y,
          but only if x is not a stopword.
    '''
    nonstop = {} # Note, there must create new dict to hold the nonstop 
    for class_type in frequency.keys():
        nonstop[class_type] = Counter(frequency[class_type]) # else the freq will be modified
        for word in stopwords:
                del nonstop[class_type][word]
    return nonstop

def laplace_smoothing(nonstop, smoothness):
    '''
    Parameters:
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of x in y, if x not a stopword
    smoothness (float)
        - smoothness = Laplace smoothing hyperparameter

    Output:
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
        - likelihood[y]['OOV'] = likelihood of an out-of-vocabulary word given y

    Be careful that your vocabulary only counts words that occurred at least once
    in the training data for class y.
    '''
    likelihood = {}
    
    for class_type in nonstop.keys():
        likelihood[class_type] = {}
        S= sum(nonstop[class_type].values())
        NumberofWords = len( nonstop[class_type].keys() )

        for word in nonstop[class_type]:
            # Note, I wrongly recoginzed number of words before
            numerator = (nonstop[class_type][word] + smoothness) 
            denominator = smoothness *(NumberofWords + 1) + S
            likelihood[class_type][word] = float(numerator) / float( denominator )
        likelihood[class_type]['OOV'] = smoothness / float(denominator)

    return likelihood

def naive_bayes(texts, likelihood, prior):
    '''
    Parameters:
    texts (list of lists) -
        - texts[i][k] = k'th token of i'th text
    likelihood (dict of dicts) 
        - likelihood[y][x] = Laplace-smoothed likelihood of x given y
    prior (float)
        - prior = the prior probability of the class called "pos"

    Output:
    hypotheses (list)
        - hypotheses[i] = class label for the i'th text
    '''
    hypotheses = []
   
    for text in texts:
        logProbability = {'pos': np.log(prior),'neg':np.log(1- prior)}
        for word in text:
            if word in stopwords:
                continue
            for y in logProbability.keys():
                if word not in likelihood[y].keys():
                    logProbability[y] += np.log(likelihood[y]['OOV'])
                else:
                    logProbability[y] += np.log(likelihood[y][word])
            
                
                #
        if logProbability['pos'] >= logProbability["neg"]:
            hypotheses.append( 'pos')
        else:
            hypotheses.append('neg')
        
            
    return hypotheses



def optimize_hyperparameters(texts, labels, nonstop, priors, smoothnesses):
    '''
    Parameters:
    texts (list of lists) - dev set texts
        - texts[i][k] = k'th token of i'th text
    labels (list) - dev set labels
        - labels[i] = class label of i'th text
    nonstop (dict of Counters) 
        - nonstop[y][x] = frequency of word x in class y, x not stopword
    priors (list)
        - a list of different possible values of the prior
    smoothnesses (list)
        - a list of different possible values of the smoothness

    Output:
    accuracies (numpy array, shape = len(priors) x len(smoothnesses))
        - accuracies[m,n] = dev set accuracy achieved using the
          m'th candidate prior and the n'th candidate smoothness
    '''
    accuracies = np.zeros( (len(priors),len(smoothnesses)) )
    for m in range(len(priors)):
        for n in range(len(smoothnesses)):
            likelihood = laplace_smoothing(nonstop,smoothnesses[n])
            hypotheses = naive_bayes(texts, likelihood, priors[m])
            # Cite from the notebook, calculate the accuracy
            count_correct = 0
            for (y,yhat) in zip(labels, hypotheses):
                if y==yhat:
                    count_correct += 1
                
            accuracies[m,n] = count_correct / len(labels)
    return accuracies
                          
