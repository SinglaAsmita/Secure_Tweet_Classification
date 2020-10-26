# Import Statements
import sys
import string
import hashlib
import numpy as np
import pandas as pd
from time import time

import nltk
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Root directories path
ROOT_DIR = '/home/asmita/Desktop/Capstone/Capstone20/'
DATA61_ROOT = "/home/asmita/Desktop/Capstone/mp-spdz-0.1.9/Player-Data/"


def pre_process_basic(tweet_text):
    """ Pre-process Text data basic text cleaning.

    :param tweet_text: string
    :return tweet_text: string """
    try:
        tweet_text = tweet_text.lower()
        tweet_text = tweet_text.replace('<br />', ' ')
        tweet_text = tweet_text.translate(str.maketrans(' ', ' ', string.punctuation))
        tweet_text = pre_process_advance(tweet_text)
    except:
        print("Some unexpected error occurred:", sys.exc_info()[1])
    return tweet_text


def pre_process_advance(tweet_text):
    """ Pre-process Text data advance pre-processing.

    :param tweet_text: string
    :return tweet_text: string """
    try:
        lemmatizer = WordNetLemmatizer()
        tweet_text = [lemmatizer.lemmatize(token) for token in tweet_text.split(" ")]
        tweet_text = [lemmatizer.lemmatize(token, "v") for token in tweet_text]
        stop_words = set(stopwords.words('english'))
        tweet_text = [word for word in tweet_text if not word in stop_words]
        tweet_text = nltk.pos_tag(tweet_text)
        tweet_text = ' '.join([word + '/' + pos for word, pos in tweet_text])
    except:
        print("Some unexpected error occurred:", sys.exc_info()[1])
        exit(2)
    return tweet_text


def get_hash_value(p, a, b, N):
    """ Create a hash value using hash function ((a · N + b) mod p) mod 2l  1.

    :param p, a, b, N: int
    :return bit_string: int """
    # l is length of bit_string
    l = 230
    bit_string = ((a * N + b) % p) % (2 ** l) - 1
    return bit_string


def create_bitstring_sha224(imp_features):
    """ Create bit string using SHA 224.

    :param imp_features: string
    :return feature_bitstring: string """
    p = 1301081
    a = 972
    b = 52097
    sha_result = hashlib.sha224(imp_features.encode())
    hash_result = sha_result.hexdigest()
    N = int(hash_result, 16)
    feature_bitstring = get_hash_value(p, a, b, N)
    return str(feature_bitstring)


# Read and process training data
train_df = pd.read_csv(ROOT_DIR + 'training_v2.tsv', sep='\t')
train_df = train_df.dropna()
train_df['ProcessedTweet'] = train_df.tweet_text.apply(lambda tweet: pre_process_basic(tweet))

# Read and process validation data
val_df = pd.read_csv(ROOT_DIR + 'dev_v2.tsv', sep='\t')
val_df = val_df.dropna()
val_df['ProcessedTweet'] = val_df.tweet_text.apply(lambda tweet: pre_process_basic(tweet))

# Read and process test data
test_df = pd.read_csv(ROOT_DIR + 'test-gold.tsv', sep='\t')
test_df['ProcessedTweet'] = test_df.tweet_text.apply(lambda tweet: pre_process_basic(tweet))

# Create TF-IDF vectorizer
tf_idf_vector = TfidfVectorizer(ngram_range=(1, 1), binary=False, norm=None, use_idf=True, max_features=200)

# Prepare the training and val data
X_train = tf_idf_vector.fit_transform(train_df['ProcessedTweet'])
Y_train = train_df['check_worthiness']
X_val = tf_idf_vector.transform(val_df['ProcessedTweet'])
Y_val = val_df['check_worthiness']

# Get feature names and IDF values from vectorizer
features_names = tf_idf_vector.get_feature_names()
idf_values = tf_idf_vector.idf_

# Train the SVM model using only most important features
SVC_model = SVC(C=1, gamma=0.75, kernel='linear', random_state=0)
SVC_model.fit(X_train, Y_train)

coefficients = SVC_model.coef_
intercept = SVC_model.intercept_

# Extract and save the bit string of uni-grams and bi-grams to player data file using SHA224
bob_mapped_num = []
for bob_features in features_names:
    mapped_num = create_bitstring_sha224(bob_features)
    bob_mapped_num.append(mapped_num)

# Save Bob's feature data in the MP-SPDZ Player file
file_bob = open(DATA61_ROOT + "Input-P1-0", "w")
count_bob = 0
for feature in bob_mapped_num:
    file_bob.write(feature + " ")
    count_bob += 1
print("Bob's total features: ", count_bob)

# Save Bob's IDF values in the MP-SPDZ Player file
count_bob = 0
for index, val in enumerate(features_names):
    idf_val = idf_values[index]
    file_bob.write(str(idf_val) + " ")
    count_bob += 1
print("IDF count:", count_bob)

# Save Bob's classifier's coefficient values in the MP-SPDZ Player file
coefficient_arr = coefficients.toarray()
count_bob = 0
for coefficient in np.nditer(coefficient_arr):
    file_bob.write(str(coefficient) + " ")
    count_bob += 1
print("IDF count:", count_bob)

# Save Bob's classifier's intercept values in the MP-SPDZ Player file
file_bob.write(str(intercept[0]))
file_bob.close()

# Create TF-IDF Vector using Alice's text to extract Uni-grams and Bi-grams
tfidf_vector_alice = TfidfVectorizer(ngram_range=(1, 1))
Alice_feat = tfidf_vector_alice.fit_transform(val_df['ProcessedTweet'])

alice_list = []
alice_text = (val_df.iloc[0][6]).split()

count = 0
alice_mapped_num_list = []
for word in alice_text:
    alice_mapped_num = create_bitstring_sha224(word)
    alice_mapped_num_list.append(alice_mapped_num)
    count = count + 1

print("Alice's total features: ", count)

file1_alice = open(DATA61_ROOT + "Input-P0-0", "w")
for feature in alice_mapped_num_list:
    file1_alice.write(feature + " ")
file1_alice.close()

# Predict the label and get the accuracy using 5-fold cross validation
y_predicted = SVC_model.predict(X_val)
print("Predicted_label: ", y_predicted)
predicted_dist = SVC_model.decision_function(X_val)
print("Predicted_distance: ", predicted_dist)
mean_accuracy = cross_val_score(SVC_model, X_train, Y_train, scoring='accuracy', cv=10).mean()
print("mean_cross_val", mean_accuracy)
print("Accuracy: %.2f" % accuracy_score(Y_val, y_predicted))
