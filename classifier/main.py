#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import os
import platform
import json
import random
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score, precision_recall_curve, auc


# -------------------------
# Get Reproducible Results
seed = 42
os.environ['PYTHONHASHSEED']=str(seed)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras import backend as K
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
# -------------------------

# Plot for log
import matplotlib.pyplot as plt

# Deep learning librairies
from keras.utils import plot_model
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model

# Exaplainer librairies
from analyzer.lime import LimeExplainer
from analyzer.embedding import computeEmbedding
from analyzer.tds import computeTDS
from analyzer.attention import computeAttention

# Model dependencies
from preprocess.general import PreProcessing
from classifier.models.cnn_model import CNN_Classifier
from classifier.models.rnn_model import RNN_Classifier
from classifier.models.tf_model import TF_Classifier
from classifier.models.hyb_model import HYB_Classifier

# ------------------------------
# Visualization tools
# ------------------------------
def plot_history(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_loss'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model loss and accuracy')
	plt.ylabel('Loss/Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'train_acc', 'val_loss', 'val_acc'], loc='upper right')
	plt.savefig(os.path.join('results', "accuracy.png"))
	plt.show()

# ------------------------------
# TRAIN
# ------------------------------
def train(train_file, test_file,val_file, model_file, config):
	
	# preprocess data
	preprocessing = PreProcessing(model_file, config)
	# train
	train_channel_texts, train_labels, train_raw_texts = preprocessing.loadData(train_file)
	x_train, y_train , train_str= preprocessing.encodeData(forTraining=True, channel_text=train_channel_texts, labels=train_labels, predict = False)

	#test
	test_channel_texts, test_labels, test_raw_texts = preprocessing.loadData(test_file)
	x_test, y_test, test_str = preprocessing.encodeData(forTraining=False, channel_text=test_channel_texts, labels=test_labels, predict=False)
	
	#val
	val_channel_texts, val_labels, val_raw_texts = preprocessing.loadData(val_file)
	x_val, y_val, val_str = preprocessing.encodeData(forTraining=False, channel_text=val_channel_texts, labels=val_labels, predict = False)

	config = preprocessing.config # update config based on metadatas
	
	preprocessing.embedding_matrix = None
	
	# Establish params
	config["vocab_size"] = []
	for dictionary in preprocessing.dictionaries:
		config["vocab_size"] += [len(dictionary["word_index"])]
	if config['vocab_size_force'][0] > 0 :
		config["vocab_size"][0] = config['vocab_size_force']
	# save train test and val into reprocessing object
	preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test = x_train, y_train, x_val, y_val, x_test, y_test
	preprocessing.x_train_str, preprocessing.x_test_str, preprocessing.x_val_str = train_str, test_str, val_str
	
	print("Available samples:")
	print("train:", len(x_train[0]), "valid:", len(x_val[0]), "test:", len(x_test[0]))
	unique, counts = np.unique(y_train, return_counts=True)
	print("label distribution in train: ", counts)
	
	print("Train sample example:")
	for channel in range(config["nb_channels"]):
		for t, test in enumerate(x_train[channel][19:20]):
			print(config["CLASSES"][list(y_train[t]).index(1)], end=" : ")
			for token in test:
				print(preprocessing.dictionaries[channel]["index_word"].get(token, "PAD"), end=" ")
			print()

	checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=30, verbose=1, mode='min')
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, verbose=1, epsilon=1e-4, mode='min')
	callbacks_list = [checkpoint, earlystop, reduce_lr]

	# create classifier
	model_type = {"CNN" : CNN_Classifier, "RNN": RNN_Classifier, "TF": TF_Classifier, "HYB": HYB_Classifier}
	classifier = model_type[config["MODEL"]](model_file=model_file, config=config, w2vec_weights=preprocessing.embedding_matrix)
	print("-"*50)
	print("TRAIN CLASSIFIER")
	print("-"*65)
	classifier.train(x_train, y_train, x_val, y_val, callbacks_list)
	#classifier.model.save(model_file)

	# ------------------------------------
	print("-"*50)
	print("TESTING")
	print("-"*50)

	#print(preprocessing.x_test_str)
	# Save Test data set
	if x_test[0].tolist() != []:
		test_sequences = open(model_file + ".test", "w")
		for s, sequence in enumerate(x_test[0]):
			test_sequence = []
			for w in range(len(sequence)):
				word_str = preprocessing.x_test_str[0][s][w]
				test_sequence += [word_str]
			label = config["CLASSES"][list(y_test[s]).index(1)]
			if test_sequence != []:
				test_sequences.write("__" + label + "__ " + " ".join(test_sequence) + "\n")
	else:
		x_test = x_val
		y_test = y_val
	
	

	# Compute test score
	if config["MODEL"] == "HYB":
		for channel in range(config["nb_channels"]):
			print("CHANNEL", channel)
			pre_model_file = model_file+".pre"+str(channel)
			model = load_model(pre_model_file)	
			model.evaluate(x_test[channel], y_test, verbose=1)
			# print AUC 
			#pred = model.predict(x_test)[:,1]
			#fpr, tpr, thresholds_keras = roc_curve(np.argmax(y_test,axis=1), pred)
			#AUC = auc(fpr, tpr)
			#print(f"AUC on test is {AUC}")
	else :
		# print AUC 
		pred = model.predict(x_test)[:,1]
	
		fpr, tpr, thresholds_keras = roc_curve(np.argmax(y_test,axis=1), pred)
		AUC = auc(fpr, tpr)
		print(f"AUC on test is {AUC}")

	model = load_model(model_file)	
	scores = model.evaluate(x_test, y_test, verbose=1)

	

	# ------------------------------------
	# Compute embedding explainer
	computeEmbedding(config, preprocessing, model, x_train, model_file)

	# ------------------------------------
	# Plot training & validation loss values
	if "plot" in config.keys():
		plot_history(classifier.history)

	config["loss"] = scores[0]*100 # Loss
	config["acc"] = scores[1]*100 # Accuracy

	return config

# ------------------------------
# PREDICT
# ------------------------------
def predict(text_file, model_file, config):

	# ------------------------------------------
	# Force to use CPU (no need GPU on predict)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

	# ------------------------------------------
	# Preprocess data 
	preprocessing = PreProcessing(model_file, config)
	channel_texts, _, raw_texts = preprocessing.loadData(text_file)
	x_data, _ ,_ = preprocessing.encodeData(forTraining=False, channel_text=channel_texts, labels=None)

	
	classifier = load_model(model_file)

	print("Predict sample example:")
	for channel in range(config["nb_channels"]):
		for token in x_data[channel][0]:
			print(preprocessing.dictionaries[channel]["index_word"].get(token, "PAD"), end=" ")
		print()

	# ------------------------------------------
	print("-"*50)
	print("PREDICTION")
	print("-"*50)
	
	# SOFTMAX BREAKDOWN
	if config["SOFTMAX_BREAKDOWN"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
		classifier = Model(inputs=classifier.input, outputs=layer_outputs)
		predictions = classifier.predict(x_data)[-1]
	else:
		predictions = classifier.predict(x_data)

	# Predictions
	predictions_list = []
	for p, prediction in enumerate(predictions):
		prediction = prediction.tolist()
		classe_value = max(prediction)
		classe_id = prediction.index(classe_value)
		classe_name = config["CLASSES"][classe_id]
		predictions_list += [(classe_name, classe_id)]
		print("sample", p+1, ":", classe_name, round(classe_value, 2))

	# ------------------------------------------
	# EXPLANATION
	explaners = {}
	for method in config["ANALYZER"]:

		# LOG
		print("-"*50)
		print(method)
		print("-"*50)

		# ------------------------------------------
		# TDS
		if method in ["TDS", "wTDS"]:
			model_data = []
			if config["MODEL"] == "HYB":
				tds_explainers = []
				for channel in range(config["nb_channels"]):
					pre_classifier = load_model(model_file + ".pre"+str(channel))
					pre_config = config.copy()
					pre_config["nb_channels"] = 1
					tds_explainers += [computeTDS(pre_config, preprocessing, pre_classifier, x_data[channel], predictions, weighted=method=="wTDS")]
				explaners[method] = []
				for t in range(len(preprocessing.raw_texts)):
					explainer = []
					for w in range(config["SEQUENCE_SIZE"]):
						word = []
						for channel in range(config["nb_channels"]):
							word += [tds_explainers[channel][t][w][0]]
						explainer += [word]
					explaners[method] += [explainer]

			else:
				tds_explainers = computeTDS(config, preprocessing, classifier, x_data, predictions, weighted=method=="wTDS")
				explaners[method] = tds_explainers

		# ------------------------------------------
		# LIME
		if method == "LIME":
			limeExplainer = LimeExplainer(config, preprocessing, classifier)
			lime_explainers = limeExplainer.analyze()

			fitted_explainer = []
			for t, text in enumerate(preprocessing.raw_texts):
				text_explainer = []
				explainer_dict = {}

				for classe in range(len(config["CLASSES"])):
					for word, score in lime_explainers[t].as_list(classe):
						explainer_dict[word] = explainer_dict.get(word, []) + [score]
				
				for w, word in enumerate(text.split(limeExplainer.split_expression)):
					word_explainer = []
					for c in range(config["nb_channels"]):
						word_explainer += [explainer_dict[word]]
					text_explainer += [word_explainer]
				fitted_explainer += [text_explainer]
			explaners[method] = fitted_explainer

		# ------------------------------------------
		# ATTENTION
		if method in ["ATT"]:
			att_explainers = computeAttention(config, preprocessing, classifier, x_data, predictions)
			explaners[method] = att_explainers

	# ---------------------------------------------------
	# COMPUTE RESULTS : 
	# Create json data object to transmit to the web view
	# ---------------------------------------------------
	data = {}  
	data["classifier"] = {}
	data["classifier"]["classes"] = config["CLASSES"]
	data["classifier"]["values"] = [0 for i in range(len(config["CLASSES"]))]
	data["sentences"] = []
	data["key_sentence_list"] = [[] for i in range(len(config["CLASSES"]))]
	data["global_max_TDS"] = [[0 for j in range(config["nb_channels"])] for i in range(len(config["CLASSES"]))]
	data["global_max_ATT"] = 0

	# ---------------------------------------------
	# CONSTANT FOR HTML DATA VIS
	color = ["5189c7", "e8b058", "2aa412"]
	channel_types = ["Forme", "Part-of-speech", "Lemme"]
	attention_thresold = int((config["SEQUENCE_SIZE"]*20)/100)
	#print("attention_thresold =====>", attention_thresold)
	
	# ---------------------------------------------
	# Loop on each sample
	for sentence_id in range(len(preprocessing.raw_texts)):

		print(sentence_id, "/" , len(preprocessing.raw_texts), end='\r')

		# ---------------------------------------------
		# INIT MULTIBAR GRAPH
		graph_channel_entries = []; 
		for channel in range(config["nb_channels"]):
			graph_channel_entries += ['{key : "' + channel_types[channel] + '", color : "#' + color[channel] +  '", values : [']

		# ---------------------------------------------
		# PREDICTED CLASS
		classe_value = max(predictions.tolist()[sentence_id])
		classe_id = predictions.tolist()[sentence_id].index(classe_value)
		classe_name = config["CLASSES"][classe_id]
		data["classifier"]["values"][classe_id] += 1
		
		# ---------------------------------------------
		# make key_sentence_list
		data["key_sentence_list"][classe_id] += [(sentence_id, classe_value)]
		#print(data["key_sentence_list"][classe_id])
		"""
		j = 0
		for prev_id in data["key_sentence_list"][classe_id]:
			if classe_value > max(predictions.tolist()[prev_id]):
				print(classe_value, max(predictions.tolist()[prev_id]))
				break
			j+=1
		data["key_sentence_list"][classe_id].insert(j-1, sentence_id)
		"""

		# ---------------------------------------------
		# GET EXPLAINER
		entry = []
		entry += [[]]
		entry += [predictions.tolist()[sentence_id]]
		for c in range(config["nb_channels"]):
			channel = []
			for w in range(config["SEQUENCE_SIZE"]):
				word = {}
				try:
					word_str = preprocessing.channel_texts[c][sentence_id][w]
				except:
					word_str = "UK"
				word[word_str] = {}
				for method in explaners.keys():
					word[word_str][method] = explaners[method][sentence_id][w][c]
				channel += [word]
			entry[0] += [channel]

		# ---------------------------------------------
		# LOOP ON WORDS
		# Create the html representation
		# ---------------------------------------------
		sentence = {}
		html = ['<span id="key-sentence-' + str(sentence_id) + '" class="anchor"></span>'];
		for w in range(config["SEQUENCE_SIZE"]): # loop on channel 0
			# ---------------------------------------------
			# CHECK WORD FULL-FORM
			forme = next(iter(entry[0][0][w]))
			forme = forme.replace("'", "")
			forme = forme.replace('"', '')

			# ---------------------------------------------
			# PREPARE WORD CLASSES AND ATTIBUTES
			if config["STRIDE"] == "HALF" and sentence_id != 0:
				if w > int(config["SEQUENCE_SIZE"]/2):
					# ---------------------------------------------
					# ADD HTML ANCHOR (key-passages)
					if w == int(config["SEQUENCE_SIZE"]/2)+1:
						html += ['<span id="key-sentence-' + str(sentence_id+1) + '" class="anchor"></span>'];
					key_sentence_class = "key-sentence-" + str(sentence_id) + " key-sentence-" + str(sentence_id+1);
				else:
					key_sentence_class = "key-sentence-" + str(sentence_id-1) + " key-sentence-" + str(sentence_id);
			else:
				key_sentence_class = "key-sentence-" + str(sentence_id);

			# HIDE SEPCIAL FORMS
			if forme in ["PAD", "__PARA__"]:
				key_sentence_class += " deconv_hidden"

			# ---------------------------------------------
			# LOOP ON CHANNELS
			# ---------------------------------------------
			word_data = {}
			for channel in range(config["nb_channels"]):

				# ----------------------------------------
				# GET WORD STR
				# ----------------------------------------
				word_str = next(iter(entry[0][channel][w])) 
				word_data["data-str"] = word_data.get("data-str", []) + [word_str]

				# ----------------------------------------
				# GET ATTENTION
				# ----------------------------------------
				try:
					word_att_tmp = entry[0][channel][w][word_str]["ATT"]
					word_att = []
					for i in range(attention_thresold): # KEPP ONLY THE HIGHEST VALUES
						if (max(word_att_tmp) > data.get("global_max_ATT", 0)):
							data["global_max_ATT"] = max(word_att_tmp)
						j = word_att_tmp.index(max(word_att_tmp))
						word_att += [[j, max(word_att_tmp)]]
						word_att_tmp[j] = 0
					word_data["data-att"] = word_data.get("data-att", []) + [word_att]
				except:
					pass

				# ----------------------------------------
				# GET TDS
				# ----------------------------------------
				try:
					word_tds = entry[0][channel][w][word_str]["wTDS"]
					word_data["data-tds"] = word_data.get("data-tds", []) + [word_tds]
					
					# global_max_TDS
					max_tds = max(word_tds)
					if (max_tds > data["global_max_TDS"][classe_id][channel]):
						data["global_max_TDS"][classe_id][channel] = max_tds

					# COMPUTE TDS GRAPH ENTRY
					graph_channel_entries[channel] += '{x : "' + forme + "_" + str(w) + '", y : ' + str(word_tds[classe_id]) + '},'
				except:
					pass

			# ----------------------------------------
			# ADD HTML WORD
			# ----------------------------------------
			deconv_word = "<span class='deconv_word " + key_sentence_class + "' "
			for attribute, value in word_data.items():
				deconv_word += attribute + "='" + json.dumps(value).replace("'", "&apos;") + "' "
			deconv_word += ">"
			deconv_word += "</span>"
			html += [deconv_word]

		# ----------------------------------------
		# COMPILE SENTENCE ENTRY
		# ----------------------------------------
		sentence["html"] = html
		sentence["graph"] = '['
		for channel in range(config["nb_channels"]):
			sentence["graph"] += graph_channel_entries[channel].strip(",") + "]},"
		sentence["graph"] = sentence["graph"].strip(",") + "]"
		data["sentences"] += [sentence]

	# FILL CLASSIFIER GLOBAL SCORE
	data["classifier"]["classe-value"] = {}
	for c, classe in enumerate(config["CLASSES"]):
		data["classifier"]["values"][c] = (data["classifier"]["values"][c]/(sentence_id+1))*100
		data["classifier"]["classe-value"][classe] = data["classifier"]["values"][c]

	# ---------------------------------------------
	# key_sentence_list Ordering
	for classe_id, key_sentence_list in enumerate(data["key_sentence_list"]):
		data["key_sentence_list"][classe_id] = [k[0] for k in sorted(key_sentence_list, key=lambda a: a[1], reverse=True)]
	#print(data["key_sentence_list"])
	#print(data["global_max_TDS"])

	# GET DEFAUT SELECTED CLASS (HIGHEST SCORE)
	data["classifier"]["selected_class"] = data["classifier"]["values"].index(max(data["classifier"]["values"]))

	# Add config to result
	data["config"] = config

	return data