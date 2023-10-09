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

import tensorflow as tf

from keras.models import Sequential,Model
from tensorflow.keras import layers
from keras.utils import np_utils
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import BatchNormalization, SpatialDropout1D, Convolution1D,Dropout,GlobalAveragePooling1D, AveragePooling1D

from tensorflow.python.client import device_lib

from classifier.models.abstract_model import Abstract_Classifier

class CNN_Classifier(Abstract_Classifier):

	def __init__(self, model_file, config, w2vec_weights, selected_channel=None):
		print("SELECTED CHANNEL =", selected_channel)
		self.selected_channel = selected_channel
		super().__init__(model_file, config, w2vec_weights)
	
	def getModel(self):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)
		
		# ---------------------------------------
		# MULTI CHANNELS CONVOLUTION
		# ---------------------------------------
		nb_channels = self.config["nb_channels"]

		inputs = [0]*nb_channels
		embedding = [0]*nb_channels
		
		conv = [0]*nb_channels
		pool = [0]*nb_channels

		for i in range(nb_channels):
			if self.selected_channel != None and self.selected_channel != i: continue
			print("CHANNELS ", i)

			# INPUTS
			inputs[i] = layers.Input(shape=(self.config["SEQUENCE_SIZE"],), dtype='float32', name='input_'+str(i)+'_channel')
			print("input", i,  inputs[i].shape)

			# EMBEDDING
			if self.config["W2VEC"] == -1:
				weights = None
			else:
				weights=[weight[i]]
			embedding[i] = layers.Embedding(
				self.config["vocab_size"][i],
				self.config["EMBEDDING_DIM"][i],
				input_length=self.config["SEQUENCE_SIZE"],
				weights=self.w2vec_weights,
				trainable=self.config["EMBEDDING_TRAINABLE"] == 1
			)(inputs[i])
			print("Embedding", i,  embedding[i].shape)
			print(self.config["vocab_size"])
			embedding[i] = SpatialDropout1D(0.3)(embedding[i])

			last_layer = embedding[i]

			# CONVOLUTIONs
			for h, HIDDEN_SIZE in enumerate(self.config["HIDDEN_SIZE"]):

				conv[i] = layers.Convolution1D(filters=HIDDEN_SIZE[i],padding = "same" ,strides=1, kernel_size=self.config["KERNEL_SIZE"][h][i], activation='relu')(last_layer)
				print("Conv1D", i,  conv[i].shape, "kernel size:", self.config["KERNEL_SIZE"][h][i])
				conv[i] = AveragePooling1D(padding = "same",strides=1)(conv[i])
				#conv[i] = MaxPooling1D(pool_size=FILTER_SIZES, strides=1, padding='same')(conv[i])
				#print("pool", i,  conv[i].shape)
				#conv[i] = Dropout(0.3)(conv[i])
				last_layer = conv[i]


			# DECONVOLUTION
			#conv[i] = UpSampling1D(2**len(self.config["FILTER_SIZES"]))(last_layer)
			#conv[i] = Conv1D(filters=self.config["EMBEDDING_DIM"], strides=1, kernel_size=FILTER_SIZES, padding='same', kernel_initializer='normal', activation='relu')(conv[i])
			#last_layer = GlobalAveragePooling1D()(last_layer)
			# TDS 
			#conv[i] = Lambda(lambda x: K.sum(x, axis=2))(conv[i])

		# ------------------------------------		
		# APPLY THE MULTI CHANNELS ABSTRACTION
		# ------------------------------------
		if  self.selected_channel != None:
			merged = conv[self.selected_channel]
		elif nb_channels > 1:
			merged = layers.concatenate(conv)
		else:
			merged = conv[0]
		print("Late fusion layer", merged.shape)
		#flat = layers.GlobalAveragePooling1D()(merged)
		flat = layers.Flatten()(merged)

		# ------------------
		# HIDDEN DENSE LAYER
		# ------------------	
		flat = layers.Dense(self.config["DENSE_SIZE"], kernel_initializer='uniform', activation='relu', dtype = "float32")(flat)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = self.get_loss(focal = True)
		output_acivation = 'softmax'
		flat = layers.Dropout(0.3)(flat)
		output = layers.Dense(len(self.config["CLASSES"]))(flat) #, kernel_regularizer=regularizers.l1(0.05)
		output = layers.Activation(output_acivation)(output)

		print("Output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		if  self.selected_channel != None:
			inputs = inputs[self.selected_channel]
		
		self.model = Model(inputs=inputs, outputs=output)
		op = optimizers.Adam(learning_rate=self.config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		self.model.compile(optimizer=op, loss=crossentropy, metrics= self.get_metrics())

		# add weight decay
		self.add_regularisation()

		print("-"*20)
		print("MODEL READY")
		print("-"*20)
		print(self.model.summary())

		return self.model
	
	def get_metrics(self) -> list :
		"""Return the metrics list

        Returns:
            list: list of keras metrics
        """
		metrics = [
            tf.keras.metrics.AUC(name='auc', curve='PR', num_thresholds=200, from_logits = True),
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(class_id=0, name = "precision_0"),
	    	tf.keras.metrics.Recall(class_id=0, name = "recall_0"),
            tf.keras.metrics.Precision(class_id=1, name = "precision_1"),
	         
            ]
		
		return metrics
	
	def add_regularisation(self):
		for layer in self.model.layers:
			if hasattr(layer, 'kernel_regularizer'):
				layer._kernel_regularizer = regularizers.l2(0.02)

	
	def get_loss(self, focal =True):
		if focal :
			loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing = True,
                                                           alpha = 0.7, 
                                                           gamma = 2, 
                                                           from_logits = False)
		else :
			loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
		
		return loss