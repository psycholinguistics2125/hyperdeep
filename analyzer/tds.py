#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Multiply,Convolution1D, Dense, LSTM, MultiHeadAttention, GlobalAveragePooling1D, AveragePooling1D

# -----------------------------------------------------
# OUTPUT LOG : pretty print
def plot_result(data):
	fig, ax = plt.subplots()
	index = np.arange(config["SEQUENCE_SIZE"])
	bar_width = 1/config["nb_channels"]
	colors = ['r', 'g', 'b']
	words = []
	for c in range(config["nb_channels"]):
		tds_values = []
		for w, word in enumerate(data): # first sample
			if c == 0:
				words += [preprocessing.channel_texts[c][0][w]]
			tds_values += [data[w][c][predictions_list[0][1]]]
		plt.bar(index + bar_width*c, tds_values, bar_width, color=colors[c], label='Channel' + str(c))

	plt.ylabel(method)
	plt.title(method + " for " + predictions_list[0][0])
	plt.xticks(index + (bar_width*config["nb_channels"])/2, words, rotation=90)
	plt.legend()
	plt.savefig(os.path.join('results', method + ".png"))
	plt.tight_layout()
	plt.show()

# -----------------------------------------------------
# Compute TDS explainer
def computeTDS(config, preprocessing, classifier, x_data, predictions, weighted=False):

	print("-"*50)
	print("TDS EXPLAINER")
	print("-"*50)

	# results
	explainers = []

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	# -------------------------------------------
	# Inspect model layers:
	# Get convolutionnal layers and dense weights
	hidden_layers = []
	dense_weights = []
	dense_bias = []
	if config["MODEL"] == "RNN":
		tds_outputlayer = LSTM
	else:
		tds_outputlayer =  AveragePooling1D #AveragePooling1D # Conv1D; Convolution1D
	for i, layer in enumerate(classifier.layers):	
		# CONVOLUTION layers => to compute TDS
		try:
			if isinstance(layer, tds_outputlayer):
				hidden_layers += [i+1]

			# DENSE WEIGHTS layers => to compute weighted TDS
			elif type(layer) is Dense:
				dense_weights += [layer.get_weights()[0]]
				dense_bias += [layer.get_weights()[1]]
		except:
			continue

	# Split the model to get only convultional outputs
	layer_outputs = [layer.output for layer in classifier.layers[config["nb_channels"]:hidden_layers[-1]]] 
	hidden_model = Model(inputs=classifier.input, outputs=layer_outputs)
	classifier.summary()
	hidden_model.summary()
	hidden_outputs = hidden_model.predict(x_data)

	# Create an explainer for each prediction
	for p, prediction in enumerate(predictions):

		# Tds array that contain the tds scores for each word
		explainer = []

		# log
		if p%100 == 0:
			print("sample", p+1 , "/" , len(predictions))
	
		# Loop on each word
		for w in range(config["SEQUENCE_SIZE"]):
			
			# GET TDS VALUES
			word = []
			for c in range(config["nb_channels"]):

				# -----------------------------------
				# TDS CALCULATION
				# -----------------------------------
				if not weighted: # OLD VERSION (TDS)	
					tds = sum(hidden_outputs[-(c+1)][p][w])
					wtds = []
					for classe in config["CLASSES"]:
						wtds += [tds] # Fake wTDS => repeated TDS
				else:
					# NEW VERSION (wTDS)			
					# Get conv output related to the channel c, the prediction p, the word w 
					hidden_output = hidden_outputs[-(config["nb_channels"]-c)][p][w]
					
					# nb filters of the last conv layer (output size)
					nb_filters = np.size(hidden_outputs[-(config["nb_channels"]-c)], 2)

					# Get the weight vector from the first hidden layer
					from_i = c*nb_filters*config["SEQUENCE_SIZE"] # select the sequence
					from_i = from_i + (w*nb_filters) # select the word
					to_j = from_i + nb_filters # end of the vector
					weight1 = dense_weights[0][from_i:to_j,:] # get weight vector

					# Apply weight
					vec = np.dot(hidden_output, weight1)# + dense_bias[0]
					#print(hidden_output.shape, weight1.shape, vec.shape)

					# Apply relu function
					vec2 = vec * (vec>0) # RELU

					# Get the weight vector from the last hidden layer
					weight2 = dense_weights[1]

					# Apply weight
					wtds = np.dot(vec2, weight2)# + dense_bias[1]
					wtds *= 100
					wtds = wtds.tolist()

					"""
				hidden_output = hidden_outputs[-(config["nb_channels"]-c)][p][w]
     			# nb filters of the last conv layer (output size)
     			nb_filters = np.size(hidden_outputs[-(config["nb_channels"]-c)], 2)

     			# Get the weight vector from the first hidden layer
     			from_i = c*nb_filters*config["SEQUENCE_SIZE"] # select the sequence
     			from_i = from_i + (w*nb_filters) # select the word
     			to_j = from_i + nb_filters # end of the vector
     			wtds = np.dot(hidden_output, dense_weights[0][from_i:to_j,:])# + dense_bias[1]
     			wtds *= 100
     			wtds = wtds.tolist()
					"""
					
				# ADD WORD CHANNEL TDS
				word += [wtds]

			# ADD WORD ENTRY
			explainer += [word]	

		# ADD EXPLAINER ENTRY (one for each prediction)
		explainers.append(explainer)

		 # Plot first sample
		if config.get("verbose", False):
			plot_result(explainers[0])

	return explainers