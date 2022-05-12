# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:53:03 2022

@author: Forge-15S 1650
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import pandas as pd
import json
import re
import os

LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')

#%%
class ExploratoryDataAnalysis():
    def __init__(self):
        pass
    
    def remove_tags(self, data):
        '''
        This function removes all HTML tags


        Parameters
        ----------
        data : Array
            Unprocessed data with HTML tags.

        Returns
        -------
        data : Array
            Processed data without HTML tags.

        '''
        for index, text in enumerate(data):
            data[index] = re.sub('<.*?>','',text)
        return data
    
    def lower_split(self, data):
        '''
        This function converts all letters into lowercase and split into list

        Parameters
        ----------
        data : Array
            Unprocessed data in multicases letters.

        Returns
        -------
        data : List
            Processed data in lower-case, splitted.

        '''
        for index, text in enumerate(data):
            data[index] = re.sub('[^a-zA-Z]',' ',text).lower().split()
        return data
    
    def sentiment_tokenizer(self, data, token_save_path, 
                            num_words=10000, 
                            oov_token='<OOV>', prt=False):
        '''
        This function setup the list of tokenizer for the dataset

        Parameters
        ----------
        data : list
            Un-tokenised data.
        token_save_path : string
            Path to save the token file.
        num_words : TYPE, int
            Maximum words processed. The default is 10000.
        oov_token : string, optional
            DESCRIPTION. The default is '<OOV>'.
        prt : TYPE, Boolean
            To print token dictionary for quick check . The default is False.

        Returns
        -------
        data : list
            Tokenised data.

        '''
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(data)
        
        token_json = tokenizer.to_json()
        
        with open(token_save_path,'w') as json_file:
            json.dump(token_json, json_file)
        
        # observe no of words
        word_index = tokenizer.word_index
        
        if prt==True:
            print(dict(list(word_index.items())[:10]))
        
        # vectorize sequence of txt
        data = tokenizer.texts_to_sequences(data)
        
        return data
    
    def sentiment_pad_sequence(self,data,maxlen=200, padding='post', truncating='post'):
        '''
        This function set the paddings for each items in list


        Parameters
        ----------
       data : list
            Unpadded data.
        maxlen : TYPE, optional
            DESCRIPTION. The default is 200.
        padding : TYPE, optional
            DESCRIPTION. The default is 'post'.
        truncating : TYPE, optional
            DESCRIPTION. The default is 'post'.

        Returns
        -------
         data : Array
            Padded data.

        '''
        
        data = pad_sequences(data, maxlen=maxlen, padding=padding, truncating=truncating)
        return data

class ModelCreation():
    def lstm_layer(self,num_words,nb_categories,embedding_output=64, nodes=32, dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words,embedding_output))
        model.add(Bidirectional(LSTM(nodes,return_sequences=True)))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(nodes)))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories,activation='softmax'))
        model.summary()
        
        return model
    
    def simple_lstm_layer(self,num_words,nb_categories,embedding_output=64, nodes=32, dropout=0.2):
        model = Sequential()
        model.add(Embedding(num_words,embedding_output))
        model.add(LSTM(nodes,return_sequences=True))
        model.add(Dropout(dropout))
        model.add(Dense(nb_categories,activation='softmax'))
        model.summary()
        
        return model
    
class ModelEvaluation():
    def report_metrics(self,y_true,y_pred):
        print(classification_report(y_true,y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(accuracy_score(y_true,y_pred))

#%%
  
if __name__ == '__main__':
    # Load Data
    URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
    df = pd.read_csv(URL)
    review = df['review']
    sentiment = df['sentiment']
    
    #%% EDA
    eda = ExploratoryDataAnalysis()
    test = eda.remove_tags(review)
    test = eda.lower_split(test)
    test = eda.sentiment_tokenizer(test, token_save_path=TOKENIZER_PATH, prt=True)
    test = eda.sentiment_pad_sequence(test)
    
    #%% Model Creation
    
    nb_categories = len(sentiment.unique())
    mc = ModelCreation()
    model = mc.lstm_layer(10000, nb_categories)