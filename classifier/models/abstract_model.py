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
from sklearn.utils.class_weight import compute_class_weight

class Abstract_Classifier:

	def __init__(self, model_file, config, w2vec_weights):
		self.model_file = model_file
		self.config = config
		self.w2vec_weights = w2vec_weights

	def getModel(self):
		pass

	def train(self, x_train, y_train, x_val, y_val, callbacks_list):
		self.model = self.getModel()

		self.history = self.model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=self.config["NUM_EPOCHS"], batch_size=self.config["BATCH_SIZE"], callbacks=callbacks_list) #class_weight=class_weights)
	
	def get_available_gpus(self):
		local_device_protos = device_lib.list_local_devices()
		return [x.name for x in local_device_protos if x.device_type == 'GPU']