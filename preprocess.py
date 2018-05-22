import numpy as np
import nltk
import io
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import cPickle as pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with io.open(fi, 'r', encoding='cp437') as f:
			contents = f.readlines()
			for line in contents[:hm_lines]:
				all_words = word_tokenize(line.lower())
				lexicon += list(all_words)
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)
	toRet = []
	for word in w_counts:
		if 50 < w_counts[word] < 1000:
			toRet.append(word)
	with open('lexicon.pickle', 'wb') as file:
		pickle.dump(toRet, file)
	return toRet

def sample_handling(sample, lexicon, classification):
	featureset = []
	with io.open(sample, 'r', encoding='cp437') as file:
		for l in file.readlines()[:hm_lines]:
			current_word = word_tokenize(l.lower())
			current_word = [lemmatizer.lemmatize(i) for i in current_word]
			features = np.zeros(len(lexicon))
			for word in current_word:
				if word.lower() in lexicon:
					features[lexicon.index(word)] += 1
			features = list(features)
			featureset.append([features, classification])
	return featureset

def train_test_split(pos, neg, test_size = 0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling(pos, lexicon, [1,0])
	features += sample_handling(neg, lexicon, [0,1])
	random.shuffle(features)
	features = np.array(features)

	size = int(len(features) * test_size)
	train_x = list(features[:,0][:-size])
	train_y = list(features[:,1][:-size])
	test_x = list(features[:,0][-size:])
	test_y = list(features[:,1][-size:])

	return train_x, train_y, test_x, test_y

a,b,c,d = train_test_split('pos.txt', 'neg.txt')
print 'Starting Pickle'
pickled_file = open('train_test_data.pickle', 'wb')
pickle.dump([a, b, c, d], pickled_file)
print 'Done'
