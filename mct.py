#!/usr/bin/python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Intertextuality detection using Multi-channel Convolutional Transformer (MCT)
# ACL 2023 - Supplementary Material
# Created on 18 jan. 2023
# ------------------------------------------------------------------------------
import sys
import os
import json
import platform

from preprocess.general import PreProcessing
from classifier.main import train, predict
from preprocess.w2vec import create_vectors, get_most_similar

def print_help():
    print("usage: python hyperdeep.py <command> <args>\n")
    print("The commands supported by deeperbase are:\n")
    print("\tword2vec\ttrain a word2vec model")
    print("\tnn\tquery for nearest neighbors\n")
    print("\ttrain\ttrain a CNN model for sentence classification\n")
    print("\tpredict\tpredict most likely labels")
    
def print_invalidArgs_mess():
    print("Invalid argument detected!\n")

def get_args():
    args = {}
    for i in range(2, len(sys.argv[1:])+1):
        if sys.argv[i][0] == "-":
            try:
                if sys.argv[i+1][0] != "-":
                    args[sys.argv[i]] = sys.argv[i+1]
                else:
                    args[sys.argv[i]] = True    
            except:
                args[sys.argv[i]] = True
        else:
            args[i] = sys.argv[i]
    return args

if __name__ == '__main__':

    # GET COMMAND
    try:
        command = sys.argv[1]
        args = get_args()
        if command not in ["word2vec", "nn", "train", "predict"]:
            raise
    except:
        print_help()
        exit()

    # GET CONFIG FILE
    try:
        config = json.loads(open("config.json", "r").read())
    except:
        print("Error: no config file found")
        exit()

    # EXECT COMMAND
    if command == "word2vec":
        try:
            corpus_file = args["-input"]
            model_file = args["-output"]

            preprocessing = PreProcessing(model_file, config)
            preprocessing.loadData(corpus_file)
            create_vectors(preprocessing.channel_texts, model_file, config)
        except:
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print_help()
            exit()
            
    if command == "nn": # nearest neighbors
        try:
            model = args[2]
            word = args[3]
            most_similar_list = get_most_similar(word, model)
            for neighbor in most_similar_list:
                print(neighbor[0], neighbor[1])

            # save predictions in a file
            result_path = "results/" + os.path.basename(model) + ".nn"
            results = open(result_path, "w")
            results.write(json.dumps(most_similar_list))
            results.close()

        except:
            print_invalidArgs_mess()
            print("usage: python hyperdeep.py nn <model> <word>\n")
            print("\tmodel\tmodel filename")
            print("\tword\tinput word\n")
            print_help()
            exit()
            
    if command == "train":
        try:
            train_file = args["-train"]
            test_file = args["-test"]
            val_file = args["-val"]

            model_file = args["-output"]

            # TRAIN
            config = train(train_file, test_file, val_file, model_file, config)
            with open(model_file + ".config", "w") as config_file: 
                json.dump(config, config_file)
            config["verbose"] = args.get("--verbose", False)

        except:
            raise
            print_invalidArgs_mess()
            print("The following arguments are mandatory:\n")
            print("\t-input\ttraining file path")
            print("\t-output\toutput file path\n")
            print("The following arguments for training are optional:\n")
            print("\t-w2vec\tword vector representations file path\n")
            print("\t-tg\tuse tagged inputs (TreeTagger format)\n")
            print_help()
            exit()

    if command == "predict":
        try:
            model_file = args[2]
            text_file = args[3]
            config = json.loads(open(model_file + ".config", "r").read())
            config["verbose"] = args.get("--verbose", False)

            config["ANALYZER"] = []
            if "-lime" in args.keys():
                config["ANALYZER"] += ["LIME"]
            
            if "-tds" in args.keys():
                config["ANALYZER"] += ["TDS"]

            if "-wtds" in args.keys():
                config["ANALYZER"] += ["wTDS"]

            if "-att" in args.keys():
                config["ANALYZER"] += ["ATT"]

            predictions = predict(text_file, model_file, config)

            # save predictions in a file
            result_path = os.path.join('results', os.path.basename(text_file)) + ".pred"
            results = open(result_path, "w")
            results.write(json.dumps(predictions))
            results.close()

            # export results as html
            result_html = open(os.path.join('results', os.path.basename(text_file)) + ".html", "w")
            header = open(os.path.join('results', 'templates', 'header.html')).read()
            style =  open(os.path.join('css', 'hyperdeep.css')).read()
            body = open(os.path.join('results', 'templates', 'body.html')).read()
            footer = open(os.path.join('results', 'templates', 'footer.html')).read()
            hyerdeepVis =  open(os.path.join('js', 'hyperdeep.js')).read()
            result_html.write(header)
            result_html.write("<style>" + style + "</style>")
            result_html.write("</head>")
            result_html.write(body)
            result_html.write(footer)
            result_html.write("<script>" + hyerdeepVis + "</script>")
            result_html.write("<script>var data = " + json.dumps(predictions) + ";</script>")
            result_html.write("<script>$( document ).ready(predictHandler(data));</script>")
            result_html.write("</body></html>")
            result_html.close()
            try:
                print("Trying to open " + os.path.basename(text_file) + ".html...")
                if platform.system() == "Windows":
                    os.system("start " + os.path.join('results', os.path.basename(text_file)) + ".html")
                else:
                    os.system("open " + os.path.join('results', os.path.basename(text_file)) + ".html")
            except:
                raise
                print("Failed.")
                print("Open " + os.path.basename(text_file) + ".html  file manualy to see explainer")

        except:
            raise
            print_invalidArgs_mess()
            print("usage: hyperdeep predict <model> <test-data>:\n")
            print("\t<model>\tmodel file path\n")
            print("\t<vec>\tword vector representations file path\n")
            print("\t<test-data>\ttest data file path\n")
            print_help()
            exit()
