#Importing the required packages
import numpy as np
import math
import pickle
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, CuDNNGRU, Bidirectional, Input, Dropout, Add
from keras.layers import Flatten, Activation, RepeatVector, Permute, multiply, Lambda
from keras import backend as K
#from keras.layers.embeddings import Embedding
#from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
import pandas as pd
from csv import writer
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st
np.random.seed(7)

#Dictionary creation
chunkSize = 4
dict_file = open(".\\mf\\mf_dict.pkl", "rb")
dict_Prop = pickle.load(dict_file)
def cls_predict(pred, normalize=True, sample_weight=None):
    s_mean = np.mean(pred, axis=0)
    m = max(s_mean)
    s_mean = (s_mean/m)
    return(list(s_mean))

def dictionary(chunk_size):
    dataframe = pd.read_csv(".\\mf\\train_data_mf2.csv", header=None)
    dataset = dataframe.values
    seq_dataset = dataset[:,0]
    print('Creating Dictionary:')
    dict = {}
    j = 0
    for row in seq_dataset:
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key not in dict:
                dict[key] = j
                j = j + 1
    del dataframe, dataset, seq_dataset
    return(dict)

def nGram(dataset, chunk_size, dictI):
    dict1 = list()
    for j, row in enumerate(dataset):
        string = row
        dict2 = list()
        for i in range(len(string) - chunk_size + 1):
            try:
                dict2.append(dictI[string[i:i + chunk_size]])
            except:
                None
        dict1.append(dict2)   
    return(dict1)

def final_model(filename, segSize, nonOL,filter_size):
    max_seq_len = segSize - chunkSize + 1
    overlap = 50
    model_path = '.\mf\mf_main_'+str(64)+'_model_'+str(1280)+'_'+str(nonOL)+'_'+str(filter_size)+'_'+ str(segSize) +'.h5'
    main_model = load_model(model_path, compile = False)

    dataframe2 = filename
    dataset2 = dataframe2.values
    X_test = dataset2[:,0]

    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segSize/ overlap)):
            pos = math.ceil(segSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = main_model.predict(test_seg)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)

    seq_path = '.\mf\mf_seq_'+str(64)+'_model_'+str(1280)+'_'+str(nonOL)+'_'+str(filter_size)+'_'+ str(segSize) +'.h5'
    seq_model = load_model(seq_path, compile = False)

    seq_preds = seq_model.predict(c_p)

    return seq_preds
	
# Testing
def test_fun(file, threshold):
    test_preds1 = final_model(file, 200,150,6)
    test_preds2 = final_model(file, 300,250,6)
    test_preds3 = final_model(file, 400,350,6)

    test_preds = (test_preds1 + test_preds2 + test_preds3) / 3   #Average of all the predictions

    test_preds[test_preds >= threshold] = int(1)
    test_preds[test_preds < threshold] = int(0)

    return test_preds


def print_labels(x):
  if(x != 0.0):
    return x

def final_return_2_user(inputs_path):
    ##predicted results
	final_results = test_fun(inputs_path, 0.38) 
	final_results = final_results.tolist()
	go_list = pd.read_csv('.\\mf\\mf_go_list2.csv', header=None)
	go_output_list = go_list.iloc[0].tolist()
	
	final_output = list()
	for i in range(len(final_results)):
		l = list(map(lambda x,y:0 if y==1 else x, go_output_list, final_results[i]))
		final_output.append(list(filter(print_labels, l)))
	
	return final_output

def main(input_test_path):
	# CREATING DICTIONARY
	chunkSize = 4
	
	final_output = final_return_2_user(input_test_path)
	
	return final_output
	
if __name__ == "__main__":
	main(data)
	
