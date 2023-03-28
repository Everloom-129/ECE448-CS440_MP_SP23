'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np
# import utils
# define your epsilon for laplace smoothing here
epsilon = 2

def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagger = {}
    output = []
    tag_time = Counter()
    for sentence in train:
        for word,tag in sentence:
            if word not in tagger:
                tagger[word] = Counter()
            tagger[word][tag] += 1
            tag_time[tag] += 1
    # Determine the most frequent tag for each word in the training data
    # print(tagger)
    most_tag = {}
    for word, tag_counter in tagger.items():
        most_tag[word] = tag_counter.most_common(1)[0][0]
    unseen_tag = tag_time.most_common(1)[0][0]

    # new_dev = utils.strip_tags(test)
    for sentence in test:
        #TypeError: unhashable type: 'list'
        new_sentence = [(word, most_tag.get(word, unseen_tag)) for word in sentence]
        # new_sentence = []
        # for word in sentence:
        #     word_pair = (word,most_tag.get(word, unseen_tag))
        #     new_sentence.append(word_pair)
        output.append(new_sentence)
    return output

def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagger = {}
    output = []
    tag_time = Counter()
    for sentence in train:
        for word,tag in sentence:
            if word not in tagger:
                tagger[word] = Counter()
            tagger[word][tag] += 1
            tag_time[tag] += 1
    # Determine the most frequent tag for each word in the training data
    # print(tagger)
    most_tag = {}
    for word, tag_counter in tagger.items():
        most_tag[word] = tag_counter.most_common(1)[0][0]
    unseen_tag = tag_time.most_common(1)[0][0]

    # new_dev = utils.strip_tags(test)
    for sentence in test:
        #TypeError: unhashable type: 'list'
        new_sentence = [(word, most_tag.get(word, unseen_tag)) for word in sentence]
        # new_sentence = []
        # for word in sentence:
        #     word_pair = (word,most_tag.get(word, unseen_tag))
        #     new_sentence.append(word_pair)
        output.append(new_sentence)
    return output


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tagger = {}
    output = []
    tag_time = Counter()
    for sentence in train:
        for word,tag in sentence:
            if word not in tagger:
                tagger[word] = Counter()
            tagger[word][tag] += 1
            tag_time[tag] += 1
    # Determine the most frequent tag for each word in the training data
    # print(tagger)
    most_tag = {}
    for word, tag_counter in tagger.items():
        most_tag[word] = tag_counter.most_common(1)[0][0]
    unseen_tag = tag_time.most_common(1)[0][0]

    # new_dev = utils.strip_tags(test)
    for sentence in test:
        #TypeError: unhashable type: 'list'
        new_sentence = [(word, most_tag.get(word, unseen_tag)) for word in sentence]
        # new_sentence = []
        # for word in sentence:
        #     word_pair = (word,most_tag.get(word, unseen_tag))
        #     new_sentence.append(word_pair)
        output.append(new_sentence)
    return output



