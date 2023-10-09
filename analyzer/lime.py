#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import os
from lime.lime_text import LimeTextExplainer
import random
import numpy as np

class LimeExplainer:

	# ------------------------------
	# INIT
	# ------------------------------
	def __init__(self, config, preprocessing, model):
		self.model = model
		self.config = config
		self.preprocessing = preprocessing
		self.nb_channels = len(preprocessing.channel_texts.keys())
		self.split_expression = " "
		self.explainer = LimeTextExplainer(class_names=[config["CLASSES"]], split_expression=self.split_expression)

	# -----------------------------------------------------
	# OUTPUT LOG : pretty print
	def plot_result(self, data):
		# -----------------------------------------------------
		# OUTPUT LOG : pretty print on first sample
		# data.as_pyplot_figure()
		# plt.tight_layout()
		# plt.savefig(os.path.join('results', method + ".png"))
		# plt.show()
		lime_html = data.as_html() # Plot first sample
		open(os.path.join('results', "lime.html"), "w").write(lime_html)
		try:
			print("Trying to open lime.html...")
			if platform.system() == "Windows":
				os.system("start " + os.path.join('results', "lime.html"))
			else:
				os.system("open " + os.path.join('results', "lime.html"))
		except:
			print("Failed.")
			print("Open results/lime.html file manualy to see lime explainer")

	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def classifier_fn(self, text):
		
		X = []

		# MULTI-CHANNELs
		if self.nb_channels > 1:
			for channel in range(self.nb_channels):
				X += [[]]
			for t in text:
				t = t.split(" ")
				for channel in range(self.nb_channels):
					entry = []
					for i in range(self.config["SEQUENCE_SIZE"]):
						try:
							word = t[i]
							word = word.split("**")[channel]
							entry += [self.preprocessing.dictionaries[channel]["word_index"].get(word, 0)]
						except:
							entry += [0]
					X[channel] += [entry]
			for channel in range(self.nb_channels):
				X[channel] = np.asarray(X[channel])

		# MONO CHANNEL
		else:
			for t in text:
				entry = []
				for i, word in enumerate(t.split(" ")):
					entry += [self.preprocessing.dictionaries[0]["word_index"].get(word, 0)]
				X += [entry]
			X = np.asarray(X)

		P = self.model.predict(X)
		print(P)
		return P

	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def analyze(self):

		print("-"*50)
		print("LIME EXPLAINER")
		print("-"*50)

		results = []
		# Create an explainer for each text
		for t, text in enumerate(self.preprocessing.raw_texts):
			# log
			print("sample", t+1 , "/" , len(self.preprocessing.raw_texts))
			results += [self.explainer.explain_instance(text, self.classifier_fn, num_features=len(text.split(self.split_expression)), top_labels=len(self.config["CLASSES"]))]

		print(self.config.get("verbose", False))
		if self.config.get("verbose", False):
			self.plot_result(results[0])

		return results