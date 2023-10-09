#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import layers

def drop_word(p, w, x_data, config, preprocessing):
	if config["nb_channels"] > 1:
		x_data = x_data[0]
	
	form = preprocessing.channel_texts[0][p][w]
	drop = False
	try:
		pos = preprocessing.channel_texts[1][p][w].strip(":")
		drop = pos == "SENT"
		drop = drop or pos == "PUNCT"
		drop = drop or x_data[p][w] == 1 # __UK__
	except:
		pass
	drop = drop or form == "PAD"
	
	# Latin Tesserae test
	#stopWords = ["qui", "quis", "et", "sum", "in", "is", "non", "hic", "ego", "ut"]
	#drop = drop or form in stopWords
	#if  form in stopWords:
	#	print(form)
	
	return drop

# Convert the attention range to another range, maintaining ratio
# New range : [0 ; 100]
def convertAttentionRange(attention, min_attention, max_attention):
	NewMin = 0
	NewMax = 100
	return (((attention - min_attention) * (NewMax - NewMin)) / (max_attention - min_attention)) + NewMin

def computeAttention(config, preprocessing, classifier, x_data, predictions):

	# results
	explainers = []

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	# -------------------------------------------
	# Inspect model layers:
	# Get convolutionnal layers and dense weights
	"""
	classifier.summary()
	tf_layers = []
	for i, layer in enumerate(classifier.layers):	
		if isinstance(layer, layers.GlobalAveragePooling1D):
			break
		tf_layers += [i+1]
	"""

	# Split the model to get only convultional outputs
	#layer_outputs = [layer.output for layer in classifier.layers[config["nb_channels"]:tf_layers[-1]]] 
	if config["SOFTMAX_BREAKDOWN"]:
		layer_outputs = [layer.output for layer in classifier.layers[config["nb_channels"]:-3]] 
	else:
		layer_outputs = [layer.output for layer in classifier.layers[config["nb_channels"]:-4]] 

	hidden_model = Model(inputs=classifier.input, outputs=layer_outputs)
	hidden_model.summary()
	hidden_outputs = hidden_model.predict(x_data)
	hidden_outputs = np.array(hidden_outputs[-1][1])

	# Create an explainer for each prediction
	for p, prediction in enumerate(predictions):

		print(p, "/" , len(predictions), end='\r')

		# Tds array that contain the tds scores for each word
		explainer = []

		# log
		#if p%100 == 0:
		#	print("sample", p+1 , "/" , len(predictions))

		min_attention = np.min(hidden_outputs[p])
		max_attention = np.max(hidden_outputs[p])
		attentions = np.max(hidden_outputs[p], axis=0) # Max or Sum of the Multi-Head

		# n-ieme attention
		flat = attentions.flatten()
		flat.sort()
		att_thresold = flat[-100]

		#print("-"*50)
		#print(p, "attentions", attentions.shape)
		#print(attentions)
	
		# Loop on each pair
		log = []
		for w1 in range(config["SEQUENCE_SIZE"]*config["nb_channels"]):

			channel1 = int(w1/config["SEQUENCE_SIZE"])
			_w1 = w1%config["SEQUENCE_SIZE"]
			word1 = preprocessing.channel_texts[channel1][p][_w1]
			
			# Init explainer
			if drop_word(p, _w1, x_data, config, preprocessing):
				word = [[0]*config["SEQUENCE_SIZE"]*config["nb_channels"]]*config["nb_channels"]
				if channel1 == 0:
					explainer += [word]
				continue			
			elif channel1 == 0:
				word = [0]*config["nb_channels"]
			else:
				word = explainer[_w1]
			word[channel1] = []

			for w2 in range(config["SEQUENCE_SIZE"]*config["nb_channels"]):
				channel2 = int(w2/config["SEQUENCE_SIZE"])
				_w2 = w2%config["SEQUENCE_SIZE"]
				word2 = preprocessing.channel_texts[channel2][p][_w2]

				if drop_word(p, _w2, x_data, config, preprocessing): # or attentions[w1][w2] < att_thresold
					word[channel1] += [0]
				else:
					
					# Oriented
					word[channel1] += [convertAttentionRange(attentions[w1][w2], min_attention, max_attention)]
					# Cooccurrence
					#word[channel1] += [round(max([attentions[w1][w2], attentions[w2][w1]])*10,2)]
					
					# ------------------------
					# LOG
					#log += [((str(w1)+ "_" + word1 + "_" +str(channel1), str(w2)+word2+str(channel2)), attentions[w1][w2])]
					# ------------------------
		
			# ADD WORD CHANNEL TDS
			if channel1 == 0:
				explainer += [word]
			else:
				explainer[_w1] = word
		
		# ADD WORD ENTRY
		explainers += [explainer]
		
	# ------------------------
	# LOG (on last sample)
	#log.sort(key=lambda a: a[1], reverse=True)
	#sources = []
	#targets = []
	#for _tuple in log[:200]:
	#	if _tuple[0][0] == _tuple[0][1]: continue
	#	print(_tuple)
	# ------------------------

	return explainers