from keras.utils import to_categorical

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

from test import vivesti_peremennie

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn import cross_validation



path = '../data/isxodniki/train_digit_net_noneutral_.csv'

dataset = pd.read_csv(path,  index_col=0).dropna()
nGlobal = 800
X_raw = dataset['text'][:nGlobal]
ngram_schemes = [(1, 1), (1, 2)]

max_features = 200 # n-gram number
batch_size = 256


def vivesti_peremennie():
    cont = True

    while(cont):
        a = print('Enter variable or press c:')
        a = input()
        if(a!='c'):
            try:
                print(a,'=',eval(a))
            except:
                print('No such a variable', a)
        else:
            cont= False


def start_print(X_train, X_test, y_train, y_test):
    print('---Start printing dimensions---')
    print('X_train:',X_train.shape)
    print('X_test:',X_test.shape)
    print('y_train:',y_train.shape)
    print('y_train:',y_test.shape)
    print('X_train',type(X_train))
    print('y_train',type(y_train))
    print('---Ended dimensions---')


for ngram_scheme in ngram_schemes:

    print('\n\n\n n-gram sheme:', ngram_scheme)

    tfidf_vectorizer = TfidfVectorizer(analyzer = "word", ngram_range=ngram_scheme)

    vectorizers = [tfidf_vectorizer]
    vectorizers_names = ['TF-IDF Vectorizer']

    vectorizer = vectorizers[0]

    X = vectorizer.fit_transform(X_raw)
    y = dataset['sentiment'][:nGlobal].values

    #print(y[:42])

    maxlen = X.shape[1]

    seed = 7
    test_size = 0.25
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size, random_state=seed)

    X_train = X_train.toarray()
    X_test = X_test.toarray()

    #y_train = to_categorical(y_train)

    start_print(X_train, X_test, y_train, y_test)

    print(y_train)


    model = Sequential()
    model.add(Embedding(max_features, 8, input_length=maxlen))
    model.add(LSTM(4, return_sequences=True))
    model.add(LSTM(4))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


    print(X_train.mean(axis=1))
    print(X_test.mean(axis=1))

    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        nb_epoch=2, verbose=1)


    predictions = model.predict_proba(X_test)


    vivesti_peremennie()

    print(predictions[:15],y_test[:15])



    #print('Accuracy: {}'.format(roc_auc_score(y_true=y_test, y_score=predictions)))
    #print('Accuracy score:',accuracy_score(predicted, y_test))
