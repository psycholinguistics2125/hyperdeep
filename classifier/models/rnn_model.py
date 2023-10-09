#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np

from keras import optimizers
from keras import regularizers

from keras import backend as K
from keras.models import Sequential,Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, UpSampling1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from keras.utils import np_utils
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import BatchNormalization

from tensorflow.python.client import device_lib

from classifier.models.abstract_model import Abstract_Classifier

class RNN_Classifier(Abstract_Classifier):
	
	def getModel(self):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)
		
		# ---------------------------------------
		# MULTI CHANNELS CONVOLUTION
		# ---------------------------------------
		nb_channels = self.config["nb_channels"]
		inputs = [0]*nb_channels
		sent_representation = [0]*nb_channels

		for i in range(nb_channels):
			print("CHANNELS ", i)

			# INPUTS
			inputs[i] = Input(shape=(self.config["SEQUENCE_SIZE"],), dtype='int32')
			print("input", i,  inputs[i].shape)

			# EMBEDDING
			if self.config["W2VEC"] == -1:
				weights = None
			else:
				weights=[weight[i]]
			sent_representation[i] = Embedding(
				self.config["vocab_size"][i],
				self.config["EMBEDDING_DIM"][i],
				input_length=self.config["SEQUENCE_SIZE"],
				weights=self.w2vec_weights,
				trainable=self.config["EMBEDDING_TRAINABLE"] == 1
			)(inputs[i])
			print("Embedding", i,  sent_representation[i].shape)

			# ----------
			# LSTM LAYER
			# ----------
			for hidden_size in self.config["HIDDEN_SIZE"]:
				sent_representation[i] = LSTM(hidden_size[i], return_sequences=True, dropout=self.config["DROPOUT_VAL"], recurrent_dropout=self.config["DROPOUT_VAL"])(sent_representation[i])
				
				"""
				rnn = Bidirectional(GRU(hidden_size[i], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(sent_representation[i])
				print("RNN :", rnn.shape)

				# ---------------
				# ATTENTION LAYER
				# ---------------
				attention = TimeDistributed(Dense(1, activation='tanh'))(rnn) 
				print("TimeDistributed :", attention.shape)

				# reshape Attention
				attention = Flatten()(attention)
				print("Flatten :", attention.shape)
				
				attention = Activation('softmax')(attention)
				print("Activation :", attention.shape)

				# Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
				# attention = RepeatVector(self.config["LSTM_SIZE"])(attention) # NORMAL RNN
				attention = RepeatVector(hidden_size[i]*2)(attention) # BIDIRECTIONAL RNN
				print("RepeatVector :", attention.shape)
				
				attention = Permute([2, 1])(attention)
				print("Permute :", attention.shape)

				# apply the attention		
				sent_representation[i] = multiply([rnn, attention])
				print("Multiply :", sent_representation[i].shape)
				"""

		# ------------------------------------		
		# APPLY THE MULTI CHANNELS ABSTRACTION
		# ------------------------------------
		if nb_channels > 1:
			sent_representation = concatenate(sent_representation)
		else:
			sent_representation = sent_representation[0]
		print("Late fusion layer", sent_representation.shape)
				
		# ------------------
		# HIDDEN DENSE LAYER
		# ------------------	
		flat = Flatten()(sent_representation)
		hidden_dense = Dense(self.config["DENSE_SIZE"], kernel_initializer='uniform', activation='relu')(flat)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = 'categorical_crossentropy'
		output_acivation = 'softmax'

		output = Dense(len(self.config["CLASSES"]))(hidden_dense) #, kernel_regularizer=regularizers.l1(0.05)
		output = Activation(output_acivation)(output)

		print("Output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		model = Model(inputs=inputs, outputs=output)
		op = optimizers.Adam(learning_rate=self.config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=op, loss=crossentropy, metrics=['accuracy'])

		print("-"*20)
		print("MODEL READY")
		print("-"*20)
		print(model.summary())

		return model