#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow import keras

from keras import backend as K
from keras.models import Sequential,Model
from keras.utils import np_utils
from keras import optimizers
from keras import regularizers

from classifier.models.abstract_model import Abstract_Classifier

# Implement a Transformer block as a layer
# Based on "Text classification with Transformer"
# @url : https://keras.io/examples/nlp/text_classification_with_transformer/
class TransformerBlock(layers.Layer):
	
	# num_heads: Number of attention heads
	# ff_dim: Hidden layer size in feed forward network inside transformer
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, dropout=0.2)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, attention_scores = self.att(inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attention_scores

# Implement embedding layer
# Based on "Text classification with Transformer"
# @url : https://keras.io/examples/nlp/text_classification_with_transformer/
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TF_Classifier(Abstract_Classifier):
	
	def getModel(self):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)

		nb_channels = self.config["nb_channels"]
		inputs = [0]*nb_channels
		embedding = [0]*nb_channels
		for i in range(nb_channels):
			print("CHANNELS ", i)

			# ------------------
			# INPUT LAYER
			# ------------------
			inputs[i] = layers.Input(shape=(self.config["SEQUENCE_SIZE"],), dtype='int32')
			print("input", i,  inputs[i].shape)

			# ------------------
			# EMBEDDING LAYER
			# ------------------
			embedding[i] = TokenAndPositionEmbedding(self.config["vocab_size"][i], self.config["vocab_size"][i], self.config["EMBEDDING_DIM"][0])(inputs[i])
		
		embedding = embedding[0]			

		# ------------------
		# ATTENTION LAYER
		# ------------------
		transformer_block, attention_scores = TransformerBlock(embed_dim=self.config["EMBEDDING_DIM"][0], num_heads=2, ff_dim=self.config["HIDDEN_SIZE"][0][0])(embedding)
		#layer = layers.MultiHeadAttention(num_heads=self.config["KERNEL_SIZE"][0][0], key_dim=self.config["HIDDEN_SIZE"][0][0])
		#attention_layer, attention_scores = layer(embedding, embedding, return_attention_scores=True)

		# ------------------
		# VECTORIZE
		# ------------------
		flat = layers.GlobalAveragePooling1D()(transformer_block)
		#flat = layers.Flatten()(attention_layer)
		
		# ------------------
		# HIDDEN DENSE LAYER
		# ------------------	
		hidden_dense = layers.Dense(self.config["DENSE_SIZE"], kernel_initializer='uniform', activation='relu')(flat)


		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = 'categorical_crossentropy'
		output_acivation = 'softmax'

		output = layers.Dense(len(self.config["CLASSES"]))(hidden_dense) #, kernel_regularizer=regularizers.l1(0.05)
		output = layers.Activation(output_acivation)(output)

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
