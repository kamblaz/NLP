from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np


def f(x):
    return {
        1.0: 0,
        1.2: 1,
        1.3: 2,
        1.5: 3,
        1.7: 4,
        1.8: 5,
        2.0: 6,
        2.2: 7,
        2.3: 8,
        2.5: 9,
        2.7: 10,
        2.8: 11,
        3.0: 12,
        3.2: 13,
        3.3: 14,
        3.5: 15,
        3.7: 16,
        3.8: 17,
        4.0: 18,
        4.2: 19,
        4.5: 20,
        4.8: 21,
        5.0: 22,
    }[x]


def g(x):
    return {
        0: 1.0,
        1: 1.2,
        2: 1.3,
        3: 1.5,
        4: 1.7,
        5: 1.8,
        6: 2.0,
        7: 2.2,
        8: 2.3,
        9: 2.5,
        10: 2.7,
        11: 2.8,
        12: 3.0,
        13: 3.2,
        14: 3.3,
        15: 3.5,
        16: 3.7,
        17: 3.8,
        18: 4.0,
        19: 4.2,
        20: 4.5,
        21: 4.8,
        22: 5.0,
    }[x]


def classify_rate(rate):
    classified_rate = np.argmax(rate)
    maks = 0
    maks2 = 0
    temp = 0
    max_index = 0
    for i in range(0, len(rate)):
        if (i > 0 and i < len(rate) - 1):
            temp = rate[i] + 0.5 * rate[i - 1] + 0.5 * rate[i + 1]
        elif (i == 0):
            temp = rate[i] + 0.5 * rate[i + 1]
        else:
            temp = rate[i] + 0.5 * rate[i - 1]
        if (temp > maks):
            maks = temp
            max_index = i
    return max_index


def rate_single_review(user_review, model):
    print('inside')
    stop_words = set(stopwords.words('polish'))
    user_review = re.sub('<.*?>', ' ', user_review)
    user_review = re.sub('\W', ' ', user_review)
    user_review = re.sub('\s+[a-zA-Z]\s+', ' ', user_review)
    user_review = re.sub('\s+', ' ', user_review)
    word_tokens = word_tokenize(user_review)
    filtered_review = " ".join([w for w in word_tokens if w not in stop_words])

    x_test = filtered_review

    num_words = 1000
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(filtered_review)
    tokenizer.word_index
    print('xtest')
    print(x_test)
    x_test_tokens = tokenizer.texts_to_sequences(x_test)
    print('xtesttokens')
    print(x_test_tokens)
    num_tokens = [len(tokens) for tokens in x_test_tokens]
    num_tokens = np.array(num_tokens)

    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)

    pad = 'pre'

    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                               padding=pad, truncating=pad)

    # print(x_test_pad)

    y_res = []
    y_pred = model.predict(x=x_test_pad)
    for i in range(0, len(y_pred)):
        rate = classify_rate(y_pred[i])
        y_res.append(rate)
    # print(y_res)
    # print('returned value')
    print(y_res)
    print(g(y_res[0]))
    return g(y_res[0])
