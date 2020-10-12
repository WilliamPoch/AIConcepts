'''
William Sivutha Poch

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
import joblib

# Load data
df = pd.read_excel('data/datafiniti_hotel_reviews_v1_cleaned.xlsx')
reviews_df = df[['review_id','raw_text','sentiment_overall', 
                 'aspect_hotel_staff', 'sentiment_hotel_staff', 'aspect_hotel_amenities',
                 'sentiment_hotel_amenities', 'aspect_hotel_condition', 'sentiment_hotel_condition',
                 'aspect_cleanliness', 'sentiment_cleanliness']]

reviews_df = reviews_df.iloc[13336:20001]
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

# Select data that contains at least 1 aspect and 400 data points of noise
aspects = reviews_df.loc[(reviews_df['aspect_hotel_staff'] == 1) | (reviews_df['aspect_hotel_amenities'] == 1) | (reviews_df['aspect_hotel_condition'] == 1) | (reviews_df['aspect_cleanliness'] == 1)]

noise_df = reviews_df.loc[(reviews_df['aspect_hotel_staff'] == 0) & (reviews_df['aspect_hotel_amenities'] == 0) & (reviews_df['aspect_hotel_condition'] == 0) & (reviews_df['aspect_cleanliness'] == 0)]

aspects = aspects.append(noise_df[:400])

# Get labels
y_df = aspects[['aspect_hotel_staff', 'aspect_hotel_amenities',
                 'aspect_hotel_condition', 'aspect_cleanliness']]
y = y_df.values

# Train test split
X_train, X_test, y_train, y_test = train_test_split(aspects['stem_text'], y, stratify=y, test_size=0.2, random_state=42)

# Word embeddings
tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(reviews_df['stem_text'].values)
# joblib.dump(tokenizer, "tokenizer.pkl")
tokenizer = joblib.load("models/tokenizer.pkl")
maxlen = 100
vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_dim = 100

# Sequential Model
model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, 
                        input_length=maxlen,trainable=True))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(4, activation='sigmoid'))
model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
            epochs=25,
            verbose=False,
            validation_data=(X_test, y_test),
            batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# Predictions
y_pred = model.predict(X_test, batch_size=5, verbose=1)

# Threshold
y_pred[y_pred>=0.06] = 1
y_pred[y_pred<0.06] = 0

# Evaluation
from sklearn.metrics import hamming_loss
print(hamming_loss(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Save model
model.save_weights('models/aspect_model2.h5')
model_json = model.to_json()
with open('models/aspect_model2.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()
