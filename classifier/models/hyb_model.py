#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow import keras

from keras import backend as K
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras import optimizers
from keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from classifier.models.abstract_model import Abstract_Classifier
from classifier.models.cnn_model import CNN_Classifier
from classifier.models.tf_model import TokenAndPositionEmbedding
from classifier.models.tf_model import TransformerBlock

class HYB_Classifier(Abstract_Classifier):
	
	def getModel(self):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)

		# --------------------------------------
		# GET PRETRAINED CONVOLUTIONs
		# --------------------------------------
		nb_channels = self.config["nb_channels"]
		cnn_inputs = []
		cnn_outputs = []
		for channel in range(nb_channels):
			cnn_layers = []
			for i, layer in enumerate(self.pre_classifiers[channel].model.layers):
				if isinstance(layer, layers.Flatten):
					break
				layer.trainable = False	
				cnn_layers += [layer]
			cnn_input = cnn_layers[0].output
			cnn_output = cnn_layers[-1].output
			
			# --------------------------------------------------------------------------
			# Positional convolutional embedding
			# the convolutional embedding is shifted using the word position information
			# --------------------------------------------------------------------------
			pos_emb = tf.range(start=0, limit=self.config["SEQUENCE_SIZE"], delta=1)
			pos_emb = layers.Embedding(input_dim=self.config["vocab_size"][channel], output_dim=self.config["HIDDEN_SIZE"][-1][channel], trainable=False)(pos_emb)
			cnn_output = cnn_output + pos_emb

			# Create pretrained inputs/outputs
			cnn_inputs += [cnn_input]
			cnn_outputs += [cnn_output]

		# ------------------------------------		
		# APPLY THE MULTI CHANNELS ABSTRACTION
		# ------------------------------------
		if  nb_channels > 1:
			merged = layers.concatenate(cnn_outputs, axis=1)
		else:
			merged = cnn_outputs[0]
		print("Late fusion layer", merged.shape)

		# ------------------
		# ATTENTION LAYER
		# ------------------
		transformer_block, attention_scores = TransformerBlock(embed_dim=self.config["HIDDEN_SIZE"][0][0], num_heads=2, ff_dim=int(self.config["HIDDEN_SIZE"][0][0]))(merged)
		#layer = layers.MultiHeadAttention(num_heads=self.config["KERNEL_SIZE"][0][0], key_dim=self.config["HIDDEN_SIZE"][0][0], dropout=self.config["DROPOUT_VAL"])
		#attention_layer, attention_scores = layer(merged, merged, return_attention_scores=True)

		# ------------------
		# VECTORIZE
		# ------------------
		flat = layers.Flatten()(transformer_block)
		#flat = layers.GlobalAveragePooling1D()(transformer_block)

		# ------------------
		# HIDDEN DENSE LAYER
		# ------------------	
		hidden_dense = layers.Dense(self.config["DENSE_SIZE"], kernel_initializer='uniform', activation='relu')(flat)
		#hidden_dense = layers.Dropout(self.config["DROPOUT_VAL"])(hidden_dense)
		#hidden_dense = layers.Dense(int(self.config["DENSE_SIZE"]/2), kernel_initializer='uniform', activation='relu')(hidden_dense)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = self.get_loss(focal=True)
		output_acivation = 'softmax'

		output = layers.Dense(len(self.config["CLASSES"]))(hidden_dense) #, kernel_regularizer=regularizers.l1(0.05)
		output = layers.Activation(output_acivation)(output)

		print("Output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		model = Model(inputs=cnn_inputs, outputs=output)
		op = optimizers.Adam(learning_rate=self.config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=op, loss=crossentropy, metrics=self.get_metrics())

		print("-"*20)
		print("MODEL READY")
		print("-"*20)
		print(model.summary())

		return model

	def get_loss(self, focal =True):
		if focal :
			loss = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing = True,
                                                           alpha = 0.7, 
                                                           gamma = 2, 
                                                           from_logits = False)
		else :
			loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
		
		return loss

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
	
	def train(self, x_train, y_train, x_val, y_val, callbacks_list):

		nb_channels = self.config["nb_channels"]

		# -----------------
		# PRETRAINING
		# -----------------
		self.pre_classifiers = []
		for channel in range(nb_channels):
			#self.pre_classifiers += [load_model(self.model_file+".pre"+str(channel))]
			pre_model_file = self.model_file+".pre"+str(channel)
			checkpoint = ModelCheckpoint(pre_model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
			earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, verbose=1, mode='min')
			callbacks_list = [checkpoint, earlystop]
			self.pre_classifiers += [CNN_Classifier(pre_model_file, self.config, self.w2vec_weights, channel)]
			self.pre_classifiers[channel].train(x_train[channel], y_train, x_val[channel], y_val, callbacks_list)
			#self.pre_classifiers[channel].model.save(self.model_file+".pre"+str(channel))


		# -----------------
		# TRANSFORMERS TRAINING
		# -----------------
		checkpoint = ModelCheckpoint(self.model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3, verbose=1, mode='min')
		#earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')
		callbacks_list = [checkpoint, earlystop]
		self.model = self.getModel()
		self.history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.config["NUM_EPOCHS"], batch_size=self.config["BATCH_SIZE"], callbacks=callbacks_list)
