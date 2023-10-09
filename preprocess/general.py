#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import sys
import numpy as np
import os
import pickle
import re

from keras.utils import np_utils
from preprocess.w2vec import create_vectors

# ----------------------------------------
# Preprocess text from input file
# Format : 
# __LABEL1__ word1 word2 word3 ...
# __LABEL2__ word1 word2 word3 ...
# ----------------------------------------
class PreProcessing:

	# ------------------------------
	# INIT
	# ------------------------------
	def __init__(self, model_file, config):
		self.model_file = model_file
		self.config = config

	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadData(self, corpus_file):   
			
		self.corpus_file = corpus_file
		self.raw_texts = []
		self.channel_texts = {}
		self.labels = []

		f = open(corpus_file, "r")
		lines = f.readlines()

		cpt = 0
		print("-"*50)
		print("PREPROCESS SAMPLES")
		print("-"*50)

		# Update config
		self.config["CLASSES"] = self.config.get("CLASSES", [])
		#self.config["nb_channels"] = len(self.config["EMBEDDING_DIM"])
		print("NB CHANNELS:", self.config["nb_channels"])

		for line in lines:
			if "--" in line: continue

			if cpt%100 == 0:
				sys.stdout.write('%s / %s \r' % (cpt, len(lines)))

			# LABELS
			if line[:2] == "__" and line[2:8] != "PARA__":
				try:
					label = line.split("__ ")[0].replace("__", "")
					try:
						self.labels += [self.config["CLASSES"].index(label)]
					except:
						self.labels += [len(self.config["CLASSES"])]
						self.config["CLASSES"] += [label]
					line = line.replace("__" + label + "__ ", "")
				except:
					raise
					print("error with line:", line)

			self.raw_texts += [line]

			# TEXT
			sequence = []
			for c in range(self.config["nb_channels"]):
				sequence += [[]]
				words = line.split()
				for token in range(self.config["SEQUENCE_SIZE"]):
					try:
						args = words[token].split("**")
						sequence[c] += [args[c]]
					except:
						sequence[c] += ["PAD"]

			for c in range(self.config["nb_channels"]):
				self.channel_texts[c] = self.channel_texts.get(c, [])
				self.channel_texts[c].append(sequence[c])
		
			cpt += 1
		f.close()

		return self.channel_texts, self.labels, self.raw_texts

	# ---------------------------------------------
	# Encode data : Convert data to numerical array
	# forTraining=True : 
	#		- create a new dictionary
	#		- Split data (Train/Validation)
	# ---------------------------------------------
	def encodeData(self, forTraining=False, channel_text = None, labels = None, predict = True):
		if labels is not None : 
			self.labels == labels
		
		dictionaries, datas = self.loadIndex(forTraining, channel_text)

		for d in range(self.config["nb_channels"]):
			print('Found %s unique tokens in channel ' % len(dictionaries[d]["word_index"]), d+1)
		
		indices = np.arange(datas[0].shape[0])
		x_str = []		
		for c, data in enumerate(datas):
			data = data[indices]
			str_ = []
			for i in indices:
				str_ += [channel_text[c][i]]
			x_str +=[str_]

		if not predict :
			self.labels = np_utils.to_categorical(np.asarray(self.labels))
		else : 
			self.labels = None
			self.dictionaries = dictionaries
	


		if forTraining : 
			self.dictionaries = dictionaries

		

		return datas, self.labels, x_str
	# ----------------------------------------
	# Train Word2Vec embeddings
	# ----------------------------------------
	def loadEmbeddings(self):

		# CREATE WORD2VEC VECTORS
		create_vectors(self.channel_texts, self.model_file, self.config)

		# Make embedding_matrix from vectors
		self.embedding_matrix = []
		for i in range(self.config["nb_channels"]):
			my_dictionary = self.dictionaries[i]["word_index"]
			embeddings_index = {}
			vectors = open(self.model_file + ".word2vec" + str(i) ,'r')
			
			for line in vectors.readlines():
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs

			print('Found %s word vectors.' % len(embeddings_index))
			self.embedding_matrix += [np.zeros((len(my_dictionary), self.config["EMBEDDING_DIM"][i]))]
			for word, j in my_dictionary.items():
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					# words not found in embedding index will be all-zeros.
					self.embedding_matrix[i][j] = embedding_vector
			vectors.close()

	# ----------------------------------------
	# LoadIndex
	# CREATE WORD INDEX DICTIONARY
	# ----------------------------------------
	def loadIndex(self, forTraining, channel_text = None):
		if channel_text is not None : 
			self.channel_texts == channel_text

		if forTraining:
			print("CREATE A NEW DICTIONARY")
			dictionaries = []
			indexes = [1,1,1]
			for i in range(3):
				dictionary = {}
				dictionary["word_index"] = {}
				dictionary["index_word"] = {}
				dictionary["word_index"]["PAD"] = 0  # Padding
				dictionary["index_word"][0] = "PAD"
				dictionary["word_index"]["__UK__"] = 1 # Unknown word
				dictionary["index_word"][1] = "__UK__" 
				dictionaries += [dictionary]
		else:
			with open(self.model_file + ".index", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", self.model_file + ".index")
				dictionaries = pickle.load(handle)
		datas = []		

		for channel, text in self.channel_texts.items():
			datas += [(np.zeros((len(text), self.config["SEQUENCE_SIZE"]))).astype('int32')]	

			line_number = 0
			for i, line in enumerate(text):
				
				words = line[:self.config["SEQUENCE_SIZE"]]
				sentence_length = len(words)
				sentence = []

				for j, word in enumerate(words):
					if word not in dictionaries[channel]["word_index"].keys():
						if forTraining:
							# ----------------------------
							# SKIP WORDS
							skip_word = False
							for f in self.config["FILTERS"]:
								if not f.strip(): continue
								if any(re.match(f, self.channel_texts[t][i][:self.config["SEQUENCE_SIZE"]][j]) for t in range(self.config["nb_channels"])):
									skip_word = True
									break
							if skip_word: 
								dictionaries[channel]["word_index"][word] = dictionary["word_index"]["__UK__"]
							# ----------------------------

							else:	 
								indexes[channel] += 1
								dictionaries[channel]["word_index"][word] = indexes[channel]
								dictionaries[channel]["index_word"][indexes[channel]] = word

						else:        
							# FOR UNKNOWN WORDS
							print("UNKNOWN", word)
							print(list(dictionaries[channel]["word_index"])[:10])
							dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["__UK__"]

					sentence.append(dictionaries[channel]["word_index"][word])

				# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
				if sentence_length < self.config["SEQUENCE_SIZE"]:
					for j in range(self.config["SEQUENCE_SIZE"] - sentence_length):
						sentence.append(dictionaries[channel]["word_index"]["PAD"])
				
				datas[channel][line_number] = sentence
				line_number += 1

		if forTraining:
			with open(self.model_file + ".index", 'wb') as handle:
				pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("VOCABULARY SIZE:", len(dictionaries[0]["index_word"]))

		return dictionaries, datas
