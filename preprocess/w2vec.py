#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import gensim
import numpy as np
import os

"""
CREATE WORD VECTORS
"""
def create_vectors(texts, model_file, config):

    print('-'*50)
    print("TRAIN WORD2VEC MODEL")
    print('-'*50)
    for i in range(config["nb_channels"]):
        print("Channel", i+1)

        text_tmp_file = model_file + ".tmp"
        f = open(text_tmp_file, "w")
        text = "" 
        for line in texts[i]:
            text += " ".join(line)
            text += "\n"
        f.write(text)
        f.flush()
        f.close()

        # USE GENSIM        
        sentences = gensim.models.word2vec.LineSentence(text_tmp_file)

        # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        model = gensim.models.Word2Vec(sentences=sentences, vector_size=config["EMBEDDING_DIM"][i], window=5, min_count=0, sg=config["W2VEC"], workers=8)
        #model = FastText(sentences=sentences, size=config["EMBEDDING_DIM"], window=config["WINDOW_SIZE"], min_count=config["MIN_COUNT"], workers=8, iter=1)

        # STORE MODEL INTO A .word2vec FILE
        f = open(model_file + ".word2vec" + str(i)  ,'w')
        vectors = []
        vector = '{} {}\n'.format(len(model.wv.index_to_key), config["EMBEDDING_DIM"][i])
        vectors.append(vector)
        f.write(vector)    
        for word in model.wv.index_to_key:
            vector = word + " " + " ".join(str(x) for x in model.wv[word]) + "\n"
            vectors.append(vector)
            f.write(vector)
        f.flush()
        f.close()
        os.system('rm -rf ' + text_tmp_file + "*" )

    """
    LOG
    """
    print("word2vec done.")
    
"""
GET W2VEC MODEL
"""
def get_w2v(vectors_file):
    return gensim.models.KeyedVectors.load_word2vec_format(vectors_file, binary=False)

"""
GET WORD VECTOR
"""
def get_vector(word, w2v):
    pass  
 
"""
FIND MOST SIMILAR WORD
"""
def get_most_similar(word, vectors_file):
    w2v = get_w2v(vectors_file)
    try:
        most_similar = w2v.most_similar(positive=[word])
    except:
        most_similar = []
    return most_similar


