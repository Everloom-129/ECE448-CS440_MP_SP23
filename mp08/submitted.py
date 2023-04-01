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
epsilon = 1e-10


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
        for word, tag in sentence:
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
    # Run baseline on test set
    for sentence in test:
        new_sentence = [(word, most_tag.get(word, unseen_tag)) for word in sentence]
        output.append(new_sentence)
    return output

# three prob's helper function:

def compute_initial_likelihoods(tag_pair_count, tag_count,alpha):
    initial_p = defaultdict(float)
    denom = alpha * (len(tag_count)) + tag_count['START']
    for tag, count in tag_pair_count['START'].items():
        likelihood = (count + alpha) / denom
        initial_p[tag] = -np.log(likelihood)
    return initial_p


def compute_transition_likelihoods(tag_pair_count, tag_count,alpha):
    transition_p = defaultdict(lambda: defaultdict(float))
    for tag, count in tag_count.items():
        denom = count + (len(tag_count)) * alpha
        for tagnxt in tag_count:
            count_tag_tagnxt = tag_pair_count[tag][tagnxt]
            likelihood = (count_tag_tagnxt + alpha) / denom
            transition_p[tag][tagnxt] = -np.log(likelihood)
    return transition_p


def compute_emission_likelihoods(tag_count, tag_word_count, word_tag_count,alpha):
    emission_p = defaultdict(lambda: defaultdict(float))
    for tag, count in tag_count.items():
        denom = count + alpha * (len(tag_word_count[tag]) + 1)
        emission_p['UNKNOWN'][tag] = -np.log(alpha / denom)
        for word in tag_word_count[tag]:
            if tag in word_tag_count[word]:
                likelihood = (word_tag_count[word][tag] + alpha) / denom
                emission_p[word][tag] = -np.log(likelihood)
    return emission_p



def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # Count occurrences of tags, tag pairs, tag/word pairs
    tag_count = Counter()
    tag_pair_count = defaultdict(Counter)
    word_tag_count = defaultdict(Counter)

    tag_word_count = defaultdict(Counter)


    for sentence in train:
        prev_tag = 'START'
        for word, tag in sentence:
            tag_count[tag] += 1
            # ('START', 'START'): 35655,('START', 'DET'): 7831,('DET', 'NOUN'): 53608,
            tag_pair_count[prev_tag][tag] += 1
            word_tag_count[word][tag] += 1
            tag_word_count[tag][word] += 1
            prev_tag = tag
    # Compute smoothed probabilities and take the log
    alpha = epsilon

    initial_p = compute_initial_likelihoods(tag_pair_count, tag_count,alpha)
    transition_p = compute_transition_likelihoods(tag_pair_count, tag_count,alpha)
    emission_p = compute_emission_likelihoods(tag_count, tag_word_count, word_tag_count,alpha)  

    # Tag the test data
    predicted = []
    for sentence in test:
        tags = trellis_backtrace(sentence, initial_p, transition_p, emission_p)
        predicted_sentence = [(word, tag) for word, tag in zip(sentence, tags)]
        predicted.append(predicted_sentence)

    return predicted



def trellis_backtrace(words, initial_p, transition_p, emission_p):
	# initialize two dictionaries to hold probabilities and backtrace information
	prob = defaultdict(lambda:defaultdict(float)) # prob[i][tag] = probability of tag at position i
	backtrace = defaultdict(lambda:defaultdict()) # backtrace[i][tag] = tag that maximizes probability at position i

	# handle the first word separately to initialize probabilities
	first = words[0]
	if first not in emission_p:
		first = 'UNKNOWN'
	for tag, pb in emission_p[first].items():
		prob[0][tag] = pb + initial_p[tag]

	# loop through remaining words and update probabilities and backtrace information
	for i, word in enumerate(words[1:], start=1):
		if word not in emission_p:
			word = 'UNKNOWN'
		for cur_tag in emission_p[word]:
			argmax_v = None
			max_v = math.inf
			for last in prob[i-1]:
				current = prob[i-1][last] + transition_p[last][cur_tag] + emission_p[word][cur_tag]
				if current < max_v:
					max_v = current
					argmax_v = last
			prob[i][cur_tag] = max_v
			backtrace[i][cur_tag] = argmax_v

	# find the tag that maximizes probability at the last position
	final = None
	max_v = math.inf
	for tag in prob[len(words)-1]:
		current = prob[len(words)-1][tag]
		if current < max_v:
			max_v = current
			final = tag

	# backtrack to find the optimal sequence of tags
	result = [final]
	last = final
	for i in range(len(words) - 1):
		idx = len(words) - 2 - i
		last = backtrace[idx+1][last]
		result.append(last)
	
	# reverse the order of the tags and return the result
	return list(reversed(result))
	


def viterbi_ec(train, test):
    '''
    Implementation for the improved viterbi tagger.
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    tag_count = Counter()
    tag_pair_count = defaultdict(Counter)
    word_tag_count = defaultdict(Counter)

    tag_word_count = defaultdict(Counter)


    for sentence in train:
        prev_tag = 'START'
        for word, tag in sentence:
            tag_count[tag] += 1
            # ('START', 'START'): 35655,('START', 'DET'): 7831,('DET', 'NOUN'): 53608,
            tag_pair_count[prev_tag][tag] += 1
            word_tag_count[word][tag] += 1
            tag_word_count[tag][word] += 1
            prev_tag = tag
    # Compute smoothed probabilities and take the log
    alpha = epsilon

    initial_p = compute_initial_likelihoods(tag_pair_count, tag_count,alpha)
    # print(initial_p)
    transition_p = compute_transition_likelihoods(tag_pair_count, tag_count,alpha)
    # print(transition_p)
    # emission_p = improved_emission(tag_count, tag_word_count, word_tag_count,alpha,transition_p)  # modify this 
    emission_p = compute_emission_likelihoods(tag_count, tag_word_count, word_tag_count,alpha)  
    # print(emission_p)
    # Tag the test data
    predicted = []
    for sentence in test:
        tags = trellis_backtrace(sentence, initial_p, transition_p, emission_p)
        predicted_sentence = [(word, tag) for word, tag in zip(sentence, tags)]
        predicted.append(predicted_sentence)

    return predicted



def improved_emission(tag_count, tag_word_count, word_tag_count,base_alpha,transition_p):
    emission_p = defaultdict(lambda: defaultdict(float))
    prev_tag = 'START'
    for tag, count in tag_count.items():
        # alpha = base_alpha 
        alpha = base_alpha * transition_p[tag]
        # alpha = base_alpha *  transition_p[prev_tag][tag]

        denom = count + alpha * (len(tag_word_count[tag]) + 1)
        emission_p['UNKNOWN'][tag] = -np.log(alpha / denom)
        for word in tag_word_count[tag]:
            if tag in word_tag_count[word]:
                likelihood = (word_tag_count[word][tag] + alpha) / denom
                emission_p[word][tag] = -np.log(likelihood)
        prev_tag = tag
    return emission_p
