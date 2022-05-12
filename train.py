# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:44:34 2022

This train.py file trains the sentiments to determine if the
 review is positive or negative

@author: Forge-15S 1650
"""
import datetime
import pandas as pd
import numpy as np
import os
from sentiment_analysis_modules import ExploratoryDataAnalysis, ModelCreation, ModelEvaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

# EDA
# 1) Load data
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

#2) Data Cleaning
eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review)
review = eda.lower_split(review)

#3) Feature Selection
#4) Data Vectorization
review = eda.sentiment_tokenizer(review,TOKENIZER_PATH)
review = eda.sentiment_pad_sequence(review)

#5) Data Preprocessing
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment,axis=-1))

nb_categories = len(np.unique(sentiment))

#split train test
X_train, X_test, y_train, y_test = train_test_split(review, sentiment, test_size=.3, random_state=123)

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))
# to know [0,1] is positvite etc

mc = ModelCreation()
num_words = 10000
model = mc.lstm_layer(num_words,nb_categories)

#%% Callbacks

log_dir = os.path.join(LOG_PATH, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile, Fitting

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=tensorboard_callback)

#%% Model Evaluation

#preallocate approach (faster)
predicted_advanced = np.empty([len(X_test),2])

for index, test in enumerate(X_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test, axis=0))

# =============================================================================
# pre-allocate memory to avoid long processing
# AVOID use append, more taxing
# Use Embedding and bidirectional for NLP (not for regression)
# =============================================================================

#%%
y_pred = np.argmax(predicted_advanced, axis=-1)
y_true = np.argmax(y_test, axis=-1)

me = ModelEvaluation()
me.report_metrics(y_true,y_pred)

#%% Deployment
model.save(MODEL_SAVE_PATH)

