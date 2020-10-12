'''
William S. Poch
'''

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, word_tokenize
import keras
import numpy
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import joblib

# Load data
df = pd.read_excel('data/datafiniti_hotel_reviews_v1_cleaned.xlsx')

df = df.iloc[13336:20001]
reviews_df = df[['review_id','raw_text','sentiment_overall', 
                 'aspect_hotel_staff', 'sentiment_hotel_staff', 'aspect_hotel_amenities',
                 'sentiment_hotel_amenities', 'aspect_hotel_condition', 'sentiment_hotel_condition',
                 'aspect_cleanliness', 'sentiment_cleanliness']]

drop = reviews_df[pd.isnull(reviews_df['sentiment_overall'])].index
reviews_df.drop(drop , inplace=True)
reviews_df = reviews_df.reset_index(drop = True) 

reviews_df['sentiment_overall'].replace('No Sentiment', np.nan, inplace=True)
drop = reviews_df[pd.isnull(reviews_df['sentiment_overall'])].index
reviews_df.drop(drop , inplace=True)
reviews_df = reviews_df.reset_index(drop = True) 

# Preprocessing
import re
reviews_df['raw_text'] = reviews_df['raw_text'].str.replace('[^\w\s]','')
reviews_df['raw_text'] = reviews_df['raw_text'].str.replace('\d+', '')
reviews_df['raw_text'] = reviews_df['raw_text'].str.lower()
reviews_df['raw_text'] = reviews_df['raw_text'].str.replace('^https?:\/\/.*[\r\n]*', '')
reviews_df['raw_text'].replace('', np.nan, inplace=True)

drop = reviews_df[pd.isnull(reviews_df['raw_text'])].index
reviews_df.drop(drop , inplace=True)
reviews_df = reviews_df.reset_index(drop = True) 

from nltk.stem import PorterStemmer
porter = PorterStemmer()

from nltk import pos_tag, word_tokenize
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

stemmed = []
for sentence in reviews_df['raw_text']:
    stemmed.append(stemSentence(sentence))
    
reviews_df['stem_text'] = stemmed

# Labels
y_df = reviews_df[['aspect_hotel_staff', 'aspect_hotel_amenities',
                 'aspect_hotel_condition', 'aspect_cleanliness']]
y = y_df.values
# Load models
json_file = open("models/aspect_model2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
aspect_model = model_from_json(loaded_model_json)
aspect_model.load_weights("models/aspect_model2.h5")

tokenizer = Tokenizer(num_words=5000)
tokenizer = joblib.load("models/tokenizer.pkl")

# Word Embedding
maxlen = 100
vocab_size = len(tokenizer.word_index) + 1

X_test = tokenizer.texts_to_sequences(reviews_df['stem_text'])

X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Predictions
y_pred = aspect_model.predict(X_test, batch_size=5, verbose=1)

# Threshold
y_pred_bool = y_pred
y_pred_bool[y_pred_bool>=0.06] = 1
y_pred_bool[y_pred_bool<0.06] = 0

# Evaluation
from sklearn.metrics import hamming_loss
print(hamming_loss(y, y_pred))

from sklearn.metrics import multilabel_confusion_matrix
print(multilabel_confusion_matrix(y, y_pred))

print(classification_report(y, y_pred))

# Add predicted class labels
predictions = [[] for _ in range(len(y_pred_bool))]

for index, values in zip(range(len(y_pred_bool)), y_pred_bool):
    if values[0] == 1:
        predictions[index].append('Staff ')
    if values[1] == 1:
        predictions[index].append('Amenities ')
    if values[2] == 1:
        predictions[index].append('Condition ')
    if values[3] == 1:
        predictions[index].append('Cleanliness ')

reviews_df['aspect'] = predictions

# Ground truth labels
y_df = reviews_df[['aspect_hotel_staff', 'aspect_hotel_amenities',
                 'aspect_hotel_condition', 'aspect_cleanliness']]
y = y_df.values

d = [[] for _ in range(len(y))]

for index, values in zip(range(len(y)), y):
    if values[0] == 1:
        d[index].append('Staff ')
    if values[1] == 1:
        d[index].append('Amenities ')
    if values[2] == 1:
        d[index].append('Condition ')
    if values[3] == 1:
        d[index].append('Cleanliness ')

reviews_df['true_aspect'] = d

# View or save results
df2 = reviews_df[['raw_text', 'aspect','true_aspect']]
# df2.to_csv('df_ground_truth.csv', encoding='utf-8', index=False)
# print(df2)