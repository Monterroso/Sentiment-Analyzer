import sys, argparse
from scipy import sparse
from sklearn import linear_model
from collections import Counter
import numpy as np
import re

from collections import deque
from copy import deepcopy


import unittest
import pdb

# #read from dictionaries
# def file_to_text(filename):
# 	data = []
# 	with open(filename, encoding="utf8") as file:
# 		for line in file:
# 			data.append(line)
# 	return data

##irregular
#irregularity
#irrelevance

# Read data from file
def load_data1(filename):
	data = []
	i = 0
	with open(filename, encoding="utf8") as file:
		for line in file:
			print(i)
			i += 1
			data.append(line)
	return data

# Implement your fancy featurization here
def fancy_featurize(text):
	features = {}

	#positives = load_data1("positive-words.txt")
	#negatives = load_data1("negative-words.txt")

	# adds bag of word representation to features
	features.update(bag_of_words(text, punctuation=False, negatives=False))

	# your code here
	#lets try it with laplace smoothing
	lap = 1

	for key in features:
		features[key] += lap

	#features.update(word_bag_group(text, 2))

	#features.update(dict_observer(text, positives, negatives))

	
	return features


def word_bag_group(text,size):

	word_bag = {}

	words = text.split(" ")

	marker = 1

	queue = []

	for word in words:

		#if a word is \n, then lets completely remove it. 

		queue.append(word)

		if word == "\n":
			queue = []
			marker = 1
		elif marker < size:
			marker += 1
		else:
		
			sumset = ""

			for el in queue:
				sumset = sumset + "#$#" + el
			#If the bag of words already contains this word, just increase the count
			if word in word_bag:
				word_bag[sumset] += 1
			#Otherwise we want to add it and set the count to one
			else:
				word_bag[sumset] = 1
			queue.pop()

	return word_bag



def dict_observer(text, positives, negatives):
	sentiment_bag = {}

	#sentiment_bag["positive"] = 0
	#sentiment_bag["negative"] = 0

	words = text.split("\n")

	for word in text:
		if word in positives:
			sentiment_bag["positive"] += 1
		if word in negatives:
			sentiment_bag["negative"] += 1

	return sentiment_bag



# Adds the bag of words representation of the text to feats
def bag_of_words(text, negatives=False,punctuation=False):

	puncs = [".", ",", ";", ":", "!", "?"]
	#puncs = ["!", "?"]
	#puncs = [".","!", "?"]

	negations = ["not", "don't", "isn't"]

	word_bag = {}
	
	# do stuff here

	sections = []
	puncsections = []
	negsections = []

	words = text.split(" ")

	#if negatives is true, we want to append negatives to all after that 
	#segement up until we get to the end of the section. 

	#if punctuation is true, what we want to do. 
	#We want to append the periods at the end of all the 
	#ones before it, all the punctuation. 

	#how about instead, we divide it into sections

	if punctuation is True or negatives is True:
		
		section = []

		newwords = []
		for word in words:
			section.append(word)

			#we check if the word has punctuation
			if any(punc in word for punc in puncs):
				sections.append(section)
				section = []

		if len(section) is not 0:
			sections.append(section)

		#Now we have our sections after this. We now we can do stuff with it

		if punctuation is True:
			for sect in sections:
				marker = sect[-1]

				#now lets loop through it and get the punctuation
				markings = [i  for i in marker if i in puncs]
				#pdb.set_trace()
				#now we loop through all the elements of that section. 
				newsection = []
				for wordloc in range(len(sect) -1):
					temp = sect[wordloc]
					for punc in markings:
						temp = temp + punc

					newsection.append(temp)

				#pdb.set_trace()
				newsection.append(marker)

				puncsections.append(newsection)
					#pdb.set_trace()
					#now the section is complete, we can add it to our building list
					#newwords.append(temp)
				#newwords.append(marker)

			
			for sect in puncsections:
				for word in sect:
					newwords.append(word)


		#pdb.set_trace()
		#now we want to go through and see what we can do given the sections. 
		if negatives is True:
			# if punctuation is True:
			# 	sections = puncsections

			newsection = []
			for sectID in range(len(sections)):
				neghit = None
				newsection = []


				#pdb.set_trace()
				#we want to check if a negative word is hit
				for wordID in range	(len(sections[sectID])):
					#now check if neghit was set to true
					#pdb.set_trace()
					if neghit is not None:
						if punctuation is True:
							newsection.append(puncsections[sectID][wordID] + neghit)
						else:
							newsection.append(sections[sectID][wordID] + neghit)

					else:
						if punctuation is True:
							newsection.append(puncsections[sectID][wordID])
						else:
							newsection.append(sections[sectID][wordID])

					if sections[sectID][wordID] in negations:

						neghit = sections[sectID][wordID]

				
				negsections.append(newsection)
				#pdb.set_trace()

			newwords = []

			for sect in negsections:
				for word in sect:
					newwords.append(word)


		#pdb.set_trace()


		
		#notice our original is unchanged, it's still just the sections. 	
		words = newwords

	# 	#Now we should be done, we can set words to be newwords
	# 	words = newwords

	for word in words:
		#If the bag of words already contains this word, just increase the count
		if word in word_bag:
			word_bag[word] += 1
		#Otherwise we want to add it and set the count to one
		else:
			word_bag[word] = 1




	return word_bag


class TestStringMethods(unittest.TestCase):
	def test_bag_of_words_default(self):

		text1 = "Hello there!"

		sol1 = {}
		sol1["Hello"] = 1
		sol1["there!"] = 1

		text2 = "gratz gratZ gratz hi. n.."

		sol2 = {}
		sol2["gratz"] = 2
		sol2["gratZ"] = 1
		sol2["hi."] = 1
		sol2["n.."] = 1


		self.assertEqual(bag_of_words(text1), sol1)
		self.assertEqual(bag_of_words(text2), sol2)

	def test_bag_of_words_punc(self):
		text1 = "Hello there!"

		sol1 = {}
		sol1["Hello!"] = 1
		sol1["there!"] = 1

		text2 = "ban it, y u fahk!? fahk!? nrg..."

		sol2 = {}
		sol2["ban,"] = 1
		sol2["fahk!?"] = 2
		sol2["it,"] = 1
		sol2["y!?"] = 1
		sol2["u!?"] = 1
		sol2["nrg..."] = 1

		text3 = "y u dumb!?"

		sol3 = {}
		sol3["y!?"] = 1
		sol3["u!?"] = 1
		sol3["dumb!?"] = 1

		self.assertEqual(bag_of_words(text3,punctuation=True), sol3)
		self.assertEqual(bag_of_words(text2,punctuation=True), sol2)
		self.assertEqual(bag_of_words(text1,punctuation=True), sol1)

	def test_bag_of_words_neg(self):
		text1 = "Hello there!"

		sol1 = {}
		sol1["Hello"] = 1
		sol1["there!"] = 1

		text2 = "gratz gratZ gratz hi. n.."

		sol2 = {}
		sol2["gratz"] = 2
		sol2["gratZ"] = 1
		sol2["hi."] = 1
		sol2["n.."] = 1

		text3 = "Hi, hi hi I am not hi hi home.. hi not"

		sol3 = {}
		sol3["Hi,"] = 1
		sol3["hi"] = 3
		sol3["I"] = 1
		sol3["am"] = 1
		sol3["not"] = 2
		sol3["hinot"] = 2
		sol3["home..not"] = 1
		


		self.assertEqual(bag_of_words(text1,negatives=True), sol1)
		self.assertEqual(bag_of_words(text2,negatives=True), sol2)
		self.assertEqual(bag_of_words(text3,negatives=True), sol3)


	def test_bag_of_words_both(self):
		text3 = "Hi, hi hi I am, not hi hi home.!? hi not!"

		sol3 = {}
		sol3["Hi,"] = 1
		sol3["hi,"] = 2
		sol3["I,"] = 1
		sol3["am,"] = 1
		sol3["not.!?"] = 1
		sol3["hi.!?not"] = 2
		sol3["home.!?not"] = 1
		sol3["hi!"] = 1
		sol3["not!"] = 1


		self.assertEqual(bag_of_words(text3,punctuation=True,negatives=True), sol3)



# regularization strength to control overfitting (values closer to 0  = stronger regularization)
L2_REGULARIZATION_STRENGTH = {"dumb_featurize": 1, "fancy_featurize": .1 }

# must observe feature at least this many times in training data to include in model
MIN_FEATURE_COUNT = {"dumb_featurize": 10,  "fancy_featurize":10 }





######################################################################
## Don't edit below this line
######################################################################


def dumb_featurize(text):
	feats = {}
	words = text.split(" ")

	for word in words:
		if word == "love" or word == "like" or word == "best":
			feats["contains_positive_word"] = 1
		if word == "hate" or word == "dislike" or word == "worst" or word == "awful":
			feats["contains_negative_word"] = 1

	return feats


class SentimentClassifier:

	def __init__(self, feature_method):
		self.feature_vocab = {}
		self.feature_method = feature_method


	# Read data from file
	def load_data(self, filename):
		data = []
		with open(filename, encoding="utf8") as file:
			for line in file:
				cols = line.split("\t")
				label = cols[0]
				text = cols[1].rstrip()

				data.append((label, text))
		return data

	# Featurize entire dataset
	def featurize(self, data):
		featurized_data = []
		for label, text in data:
			feats = self.feature_method(text)
			featurized_data.append((label, feats))
		return featurized_data

	# Read dataset and returned featurized representation as sparse matrix + label array
	def process(self, dataFile, training = False):
		data = self.load_data(dataFile)
		data = self.featurize(data)

		if training:			
			fid = 0
			feature_doc_count = Counter()
			for label, feats in data:
				for feat in feats:
					feature_doc_count[feat]+= 1

			for feat in feature_doc_count:
				if feature_doc_count[feat] >= MIN_FEATURE_COUNT[self.feature_method.__name__]:
					self.feature_vocab[feat] = fid
					fid += 1

		F = len(self.feature_vocab)
		D = len(data)
		X = sparse.dok_matrix((D, F))
		Y = np.zeros(D)
		for idx, (label, feats) in enumerate(data):
			for feat in feats:
				if feat in self.feature_vocab:
					X[idx, self.feature_vocab[feat]] = feats[feat]
			Y[idx] = 1 if label == "pos" else 0

		return X, Y

	def load_test(self, dataFile):
		data = self.load_data(dataFile)
		data = self.featurize(data)

		F = len(self.feature_vocab)
		D = len(data)
		X = sparse.dok_matrix((D, F))
		Y = np.zeros(D, dtype = int)
		for idx, (data_id, feats) in enumerate(data):
			# print (data_id)
			for feat in feats:
				if feat in self.feature_vocab:
					X[idx, self.feature_vocab[feat]] = feats[feat]
			Y[idx] = data_id

		return X, Y

	# Train model and evaluate on held-out data
	def evaluate(self, trainX, trainY, devX, devY):
		(D,F) = trainX.shape
		self.log_reg = linear_model.LogisticRegression(C = L2_REGULARIZATION_STRENGTH[self.feature_method.__name__])	
		self.log_reg.fit(trainX, trainY)
		training_accuracy = self.log_reg.score(trainX, trainY)
		development_accuracy = self.log_reg.score(devX, devY)
		print("Method: %s, Features: %s, Train accuracy: %.3f, Dev accuracy: %.3f" % (self.feature_method.__name__, F, training_accuracy, development_accuracy))
		

	# Predict labels for new data
	def predict(self, testX, idsX):
		predX = self.log_reg.predict(testX)

		out = open("%s_%s" % (self.feature_method.__name__, "predictions.csv"), "w", encoding="utf8")
		out.write("Id,Expected\n")
		for idx, data_id in enumerate(testX):
			out.write("%s,%s\n" % (idsX[idx], int(predX[idx])))
		out.close()

	# Write learned parameters to file
	def printWeights(self):
		out = open("%s_%s" % (self.feature_method.__name__, "weights.txt"), "w", encoding="utf8")
		reverseVocab = [None]*len(self.feature_vocab)
		for feat in self.feature_vocab:
			reverseVocab[self.feature_vocab[feat]] = feat

		out.write("%.5f\t__BIAS__\n" % self.log_reg.intercept_)
		for (weight, feat) in sorted(zip(self.log_reg.coef_[0], reverseVocab)):
			out.write("%.5f\t%s\n" % (weight, feat))
		out.close()


# python3 sentiment_classifier.py --train train.txt --dev dev.txt --test test.txt 
if __name__ == "__main__":

	debug = False

	if debug is True:
		unittest.main()
		exit()

	ap = argparse.ArgumentParser()
	ap.add_argument("--train", required=True)
	ap.add_argument("--dev", required=True)
	ap.add_argument("--test", required=True)	

	args = vars(ap.parse_args())

	trainingFile = args["train"]
	evaluationFile = args["dev"]
	testFile = args["test"]

	for feature_method in [dumb_featurize, fancy_featurize]:
		sentiment_classifier = SentimentClassifier(feature_method)
		trainX, trainY = sentiment_classifier.process(trainingFile, training=True)
		devX, devY = sentiment_classifier.process(evaluationFile, training=False)
		testX, idsX = sentiment_classifier.load_test(testFile)
		sentiment_classifier.evaluate(trainX, trainY, devX, devY)
		sentiment_classifier.printWeights()
		sentiment_classifier.predict(testX, idsX)
