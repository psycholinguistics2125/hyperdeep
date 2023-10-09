#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras.models import Model

def computeEmbedding(config, preprocessing, classifier, x_data, model_file):

	print("-"*50)
	print("EMBEDDING EXPLAINER")
	print("-"*50)

	# ------------------------------------
	# GET WORD EMBEDDING MODEL
	layer_outputs = [layer.output for layer in classifier.layers[config["nb_channels"]:config["nb_channels"]*2]] 
	wordembedding_model = Model(inputs=classifier.input, outputs=layer_outputs)
	print("WORD EMBEDDING MODEL")
	wordembedding_model.summary()

	# ------------------------------------
	# GET SENTENCE EMBEDDING MODEL
	layer_outputs = classifier.layers[-3].output
	sentenceembedding_model = Model(inputs=classifier.input, outputs=layer_outputs)
	print("SENTENCE EMBEDDING MODEL")
	sentenceembedding_model.summary()

	# ------------------------------------
	# GET WORD EMBEDDINGS
	x_data = []
	for c in range(config["nb_channels"]):
	#for i, vocab_size in enumerate(config["vocab_size"]):
		vocab_size = config["vocab_size"][c]
		x_entry = []
		entry = []
		for word_index in range(config["vocab_size"][0]):
			if word_index%config["SEQUENCE_SIZE"] == 0 and word_index != 0:
				x_entry.append(entry)
				entry = []
			entry += [word_index%vocab_size]

		for word_index in range(config["SEQUENCE_SIZE"]-len(entry)):
			entry += [0]
		x_entry.append(entry)
		x_data += [np.array(x_entry)]
		
	if config["nb_channels"] == 1:
		wordembedding = wordembedding_model.predict(x_data[0])
	else:
		wordembedding = wordembedding_model.predict(x_data)

	# init wordembeddings
	wordembeddings = {}
	for channel in range(len(x_data)):
		wordembeddings[channel] = {}

	# READ ALL SENTENCES (TODO: optimize this!)
	for p in range(len(x_data[channel])):
		# READ SENTENCE WORD BY WORD
		for i in range(config["SEQUENCE_SIZE"]):
			# READ EACH CHANNEL
			for channel in range(config["nb_channels"]):
				index = x_data[channel][p][i]
				word = preprocessing.dictionaries[channel]["index_word"].get(index, "PAD")

				# MUTLI CHANNEL
				if config["nb_channels"] > 1:
					wordvector = wordembedding[channel][p][i]

				# ONE CHANNEL
				else:
					wordvector = wordembedding[p][i]
				
				wordembeddings[channel][word] = wordvector

	for channel in range(config["nb_channels"]):
		f = open(model_file + ".finalvec" + str(channel) ,'w')
		vectors = []
		vector = '{} {}\n'.format(len(wordembeddings[channel].keys()), config["EMBEDDING_DIM"][channel])
		vectors.append(vector)
		f.write(vector)    
		for word, values in wordembeddings[channel].items():
			vector = word + " " + " ".join([str(f) for f in values]) + "\n"
			vectors.append(vector)
			f.write(vector)
		f.flush()
		f.close()
	print("DONE.")