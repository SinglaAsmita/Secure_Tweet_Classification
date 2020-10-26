from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

import sys
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF as NMF
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from keras.layers import Conv1D
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling1D, Activation

# Root directories path
ROOT_DIR = '/home/asmita/Desktop/Capstone/Capstone20/'

dfTweets = pd.read_csv(ROOT_DIR + 'training_v2.tsv', sep='\t')
validation_df = pd.read_csv(ROOT_DIR + 'dev_v2.tsv', sep='\t')
test_df = pd.read_csv(ROOT_DIR + 'test-gold.tsv', sep='\t')


def preProcBasic(text):
    """ Pre-process Text data basic text cleaning.

    :param text: string
    :return tweet_text: string """
    try:
        text = text.lower()
        text = text.replace('<br />', ' ')
        text = text.translate(str.maketrans(' ', ' ', string.punctuation))
        text = preProcAdvanced(text)
    except:
        print("Some unexpected error occurred:", sys.exc_info()[1])
    return text


def preProcAdvanced(text):
    """ Pre-process Text data advance pre-processing.

    :param text: string
    :return tweet_text: string """
    # rev = [word for word in rev if not word in stop_words]
    # sklearn has about 318 and includes numbers together etc, for lstm if seq of words become imp then no
    try:
        lemmatizer = WordNetLemmatizer()
        text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
        text = [lemmatizer.lemmatize(token, "v") for token in text]
        # stop_words = set(stopwords.words("english"))
        # text = [word for word in text if not word in stop_words]
        text = " ".join(text)
    except:
        print("Some unexpected error occurred:", sys.exc_info()[1])
        exit(2)
    return text


# TF-IDF Vectorizer
dfTweets['ProcessedTweet'] = dfTweets.tweet_text.apply(lambda tweet: preProcBasic(tweet))
validation_df['ProcessedTweet'] = validation_df.tweet_text.apply(lambda tweet: preProcBasic(tweet))
test_df['ProcessedTweet'] = test_df.tweet_text.apply(lambda tweet: preProcBasic(tweet))

tfidf_vector = TfidfVectorizer(ngram_range=(1, 2), max_features=None)
tfidf_BoW = tfidf_vector.fit_transform(dfTweets['ProcessedTweet'])
tfidf_BOW_val = tfidf_vector.transform(validation_df['ProcessedTweet'])

# Topic Modelling with NMF
nmf = NMF(n_components=3, init='random', random_state=0)
topics_nmf = nmf.fit_transform(tfidf_BoW)
topics_nmf_val = nmf.transform(tfidf_BOW_val)

# Training and Validation data
dfTweets = dfTweets.dropna()
validation_df = validation_df.dropna()

X_train = dfTweets['ProcessedTweet']
y_train = dfTweets['check_worthiness']

X_val = validation_df['ProcessedTweet']
y_val = validation_df['check_worthiness']

combined_df = pd.concat([dfTweets, validation_df])
combined_df = combined_df.dropna()

combined_X = combined_df['ProcessedTweet']
combined_y = combined_df['check_worthiness']

X_train_Claim, X_val_Claim, Y_train_Claim, Y_val_Claim = train_test_split(combined_X, combined_y, test_size=0.2,
                                                                          shuffle=True, random_state=42)

# CNN parameters
vocab_size = 10000
maxlen = 320
batch_size = 32
filters = 8
kernel_size = 3
hidden_dims = 16
epochs = 300
embedding_dims = 200
lr = 0.001

# Create Tokenizer
tokenizer = Tokenizer(num_words=vocab_size + 1, lower=False, oov_token='UNK')
tokenizer.fit_on_texts(combined_X)

# Text_to_Sequences
X_train_Claim = tokenizer.texts_to_sequences(X_train_Claim)
X_val_Claim = tokenizer.texts_to_sequences(X_val_Claim)

# Pad sequences
X_train_Claim = sequence.pad_sequences(X_train_Claim, maxlen=maxlen, padding='post')
X_val_Claim = sequence.pad_sequences(X_val_Claim, maxlen=maxlen, padding='post')

# Glove embedding
embeddings_index = dict()
f = open('glove.twitter.27B.200d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = pd.np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

# Create a weight matrix
embedding_matrix = pd.np.zeros((vocab_size, embedding_dims))
for word, index in tokenizer.word_index.items():
    if index > vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Train CNN
model = Sequential()
model.add(Embedding(vocab_size, embedding_dims, input_length=maxlen, weights=[embedding_matrix], trainable=False))
model.add(Dropout(0.5))
model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
model.add(GlobalAveragePooling1D())
model.add(Dense(hidden_dims, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

mc = ModelCheckpoint("./claim_model_cnn.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
callbacks_list = [mc, es]

text_model = model.fit(X_train_Claim, Y_train_Claim, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_val_Claim, Y_val_Claim), callbacks=callbacks_list)

# Visualize text_model for accuracy
plt.plot(text_model.history['accuracy'])
plt.plot(text_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Visualize text_model for loss
plt.plot(text_model.history['loss'])
plt.plot(text_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(model.summary())

X = test_df["ProcessedTweet"]
X = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen=maxlen)

y_predicted = text_model.predict(X)
print(y_predicted)
