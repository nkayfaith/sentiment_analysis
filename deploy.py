# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:20:42 2022

@author: Forge-15S 1650
"""
from sentiment_analysis_modules import ExploratoryDataAnalysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

MODEL_PATH = os.path.join(os.getcwd(),'model.h5')
JSON_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')

#%% model load

sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary

#%% tokenizer load
with open(JSON_PATH,'r') as json_file:
    loaded_tokenizer = json.load(json_file)
    
#%%
#1) Load Data
new_review = ['<br \>I think the first one hour is \
    interesting but the second half of the movie is \
    boring. This movie just wasted my precious time \
    and har earned money. This movie should be banned \
    to avoid time being wasted.<br \>']
    
#dynamic
# new_review = [input('Review about movie\n')]
       
#2) Clean Data
eda = ExploratoryDataAnalysis()
removed_tags = eda.remove_tags(new_review)
cleaned_input = eda.lower_split(removed_tags)

#3) Feature Selection
#4) Tokenization
##load into keras
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
##vectorise
new_review = loaded_tokenizer.texts_to_sequences(cleaned_input)
new_review = eda.sentiment_pad_sequence(new_review)
#%% Model Predict
outcome = sentiment_classifier.predict(np.expand_dims(new_review,axis=-1))
sentiment_dict = {1:'positive',0:'negative'}
print('this review is '+sentiment_dict[np.argmax(outcome)])

# =============================================================================
# OHE, positive [0,1], negative [1,0]
# =============================================================================

