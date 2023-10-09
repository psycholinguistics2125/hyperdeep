# Multi-channel Convolutional Transformer (MCT)
MCT is an intertextuality detection tool based on a text classifier using a Multi-channel convolutionnal neural network with Multi-Head attention.
The model should be trained on data set corresponding to potential intertext sources (corpus-driven approach). Then, the prediction must be performed on a target text suspected to have been influenced by one or more of the source texts. To analyse intertextuality, MCT can use several methods:
1) Lime : the well-known library that allows to interpret any classifier : https://arxiv.org/pdf/1602.04938.pdf
2) TDS : A convolutionnal feature extraction approach that gives an activation score for each word in the text
3) wTDS : A weighted version of TDS based on Class Activation Map (CAM) algorithm.
4) ATT : Multi-Head attention scores

# requirements:
The application uses python3.6.x
The decency list of the package is available in requirements.txt
To deploy a virtual environment from it, use the commands:
	$ python -m venv myvenv
	$ source myvenv/bin/activate
	$ pip install -r requirements.txt

# HOW TO USE IT
# data
The data is stored in the data/ folder. The training set should be splited into phrases of fixed length (50 words by default). Each phrase should have a label name at the beginning of the line. A label is written : __LABELNAME__
MCT uses multi-channel encoded data. For each word, each channel's value should be separated with ** as the folowing example:
	__propertius__ impositis**Verb**IMPONO pressit**Verb**PREMO Amor**Subs**AMOR_N pedibus**Subs**PES
MCT is distributed with an example of corpus named poetae (in data/poetae) and a test data set (data/odiv-test). The corpus represents works of seven classical Latin poets: Catullus, Horace, Juvenal, Lucretius, Propertius, Tibullus and Virgil. And the test data set represents the letter 21 from Ovid works.Texts are obtained from the L.A.S.L.A.
 
# Train classifier (Example using the demo train dataset : data/poeta)
To train the classifier:
	$ python3 mct.py train -input data/poetae -output bin/poetae
The command will create bin/poetae folder containing the trained model.

# Predict task (Example using the demo test dataset : data/odiv-test)
Predictions based on new text.
	$ python3 mct.py predict bin/poetae data/odiv-test -wtds -att
The command will create a json result file in the folder result/ (create the folder if needed).

# Intertextuality detection tool
 This json result file contains the classification of each sample as well as the feature extraction related to each selected method. To select a method, use optional flags with the command predict:
	$ python python3 mct.py predict bin/poetae data/odiv-test [-lime] [-tds] [-wtds] [-att]
-lime : correspond to the Lime application
-tds :  correspond to convolutional features extraction
-wtds : correspond to convolutional features extraction + Class Activation Map
-att : correspond to Multi-head attention features extraction

# CONFIGURATIONN FILE
The application uses a config file named config.json (in the root directory). The hyper-parameters and the general architecture can be modified using this config file instead of modifying the source code.

