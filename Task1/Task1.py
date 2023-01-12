import os
import re
from datetime import datetime

import gensim
import numpy as np
import seaborn as sns
import spacy
import torch as nn
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import preprocessing

# nltk.download('stopwords')
from nltk.corpus import stopwords

from gensim.models import KeyedVectors, Word2Vec, fasttext
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Flatten, SpatialDropout1D
import tensorflow as tf
from torch.nn import Dropout

stop_words = stopwords.words('english')  # english stopwords
word_tokenize = spacy.blank('en')


def tokenize(text, remove_stop=True):
    """
    Clean a text and return tokens
    Input: string
    Output: normalised list of tokens from text
    """
    cleaned = re.sub(r'[^\w\s_]+|\d+', '', text).strip().lower()  # remove numbers because queries don't contain
    tokenized = [str(tok).lower() for tok in word_tokenize(cleaned)]  # tokenize document
    if remove_stop:
        tokenized = [token for token in tokenized if token not in stop_words]  # remove stopwords
    tokenized = [x.strip() for x in tokenized if x.strip() != ' ']
    tokenized = [x.strip() for x in tokenized if x.strip() != '']
    return tokenized


def load_fasttext():
    ft_file = ".cache/cc.en.100.bin"
    print(f'>>> Load FastText from {ft_file}', end=' ')
    start_time = datetime.now()
    ft_cc = fasttext.load_facebook_vectors(ft_file)
    time_elapsed = datetime.now() - start_time
    print(f'| Took: {time_elapsed}')
    return ft_cc


def load_glove():
    print('>>> Load GloVe')
    word2vec_model_path = './python_materials/glove.6B.50d.txt'
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=False, unicode_errors='ignore',
                                                       encoding='utf-8')
    word2vec_model.fill_norms()
    return word2vec_model


def load_word2vec():
    saved_embeddings = './python_materials/word2vec_100.model'
    if os.path.exists(saved_embeddings):
        print('>>> Load Word2Vec')
        wv_model = Word2Vec.load(saved_embeddings).wv
    else:
        wv_model = gensim.downloader.load('word2vec-google-news-300')
    return wv_model

    # how to get the sentence(text) vector through word vector
    # 1. accumulation
    # 2. average
    # 3. TF-IDF weighted average
    # 4. ISF embedding

    # average
    # problem1: the word have not present in vector model(dictionary)
    # A: map to different random vectors (-1 - 1)


def word_embedding(model, document, length):
    vector = []
    for j in range(0, len(document)):
        sent_vector = []
        sentence = tokenize(document[j])

        if sentence[0] in model:
            sts_vec = model[sentence[0]]
            sent_vector.append(sts_vec)
        else:
            sts_vec = -1 + 2 * np.random.random(50)
            sent_vector.append(sts_vec)

        for i in range(1, length):
            if len(sentence) <= i:
                vec_new = 0 * np.random.random(50)
                sent_vector.append(vec_new)
            else:
                if sentence[i] in model:
                    vec_new = model[sentence[i]]
                    sent_vector.append(vec_new)
                else:
                    vec_new = -1 + 2 * np.random.random(50)
                    sent_vector.append(vec_new)
                    # random
        vector.append(sent_vector)
    # print('Word dict length:', len(vector), 'Max sentence size:', len(vector[0]), 'Vector dimension', len(vector[0][0]))
    # 27073 data, 256 max length, 50 dimension
    return vector


if __name__ == '__main__':
    dataset_name = './python_materials/mesh.csv'
    df = pd.read_csv(dataset_name)

    df = df.dropna()  # drop rows with null values if exist
    df_distribution = df.groupby(['label'])['label'].count()
    # print('Sample data distribution:')
    # print(df_distribution)
    content = df['Text'].values
    labels = pd.get_dummies(df['label']).values

    # labels encoding
    enc = preprocessing.LabelEncoder()

    model_glove = load_glove()
    # model_wv = load_word2vec()

    LENGTH = 64
    HIDDEN_LAYER_SIZE = 64

    dic2 = model_glove.index_to_key
    print('>>> GloVe loaded')
    print('>>> Dataset size:', len(content))
    vec = word_embedding(model_glove, content, LENGTH)
    vec = np.array(vec)

    print('>>> Word embedded')

    model = Sequential()
    # already embedded
    # model.add(Embedding(len(vec), len(vec[0][0]), input_length=LENGTH))
    model.add(SpatialDropout1D(rate=0.2))
    model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(21, activation='softmax'))

    # compile network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.build(input_shape=(vec.shape))

    # summarize defined model
    model.summary()

    # LSTM
    # model = Sequential()
    # # model.add(Dense(200, activation='relu', input_shape=(vec.shape[1],)))
    # # model.add(Dense(50, activation='relu'))
    # model.add(Dense(21, activation='softmax'))

    X_train, X_test, y_train, y_test = train_test_split(vec, labels, test_size=0.15, random_state=0)
    print('>>> Model training')
    print('    training size:', len(y_train), 'testing size:', len(y_test))
    print(X_train.shape, y_train.shape)

    batch_size = 128
    history = model.fit(X_train, y_train, epochs=20, batch_size=batch_size, verbose=2)

    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('acc')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    _, train_acc = model.evaluate(X_train, y_train, verbose=2)
    _, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('Train: %.4f, Test: %.4f' % (train_acc, test_acc))
