import sys
from keras.optimizers import Adam
from keras.layers import Dense, GRU, Embedding, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import pandas as pd
import csv
from stempel import StempelStemmer


def loadDataFromCSV(filename):
    with open(filename, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        data = list(reader)
    df = pd.read_csv(filename)
    rates = df['Rating'].tolist()
    reviews = df['Review'].tolist()
    return reviews, rates


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
        4.3: 20,
        4.5: 21,
        4.7: 22,
        4.8: 23,
        5.0: 24,
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
        20: 4.3,
        21: 4.5,
        22: 4.7,
        23: 4.8,
        24: 5.0,
    }[x]


def count_number_of_samples_for_each_class(yall):
    numberOfSamples = []
    for i in range(0, 25):
        numberOfSamples.append(yall.count(i))
    return numberOfSamples


def assign_weight(number_of_all_samples, number_of_clesses, number_of_samples_for_class):
    return number_of_all_samples / (number_of_clesses * number_of_samples_for_class)


def assign_weight_for_each_class(number_of_all_samples, number_of_samples_for_each_class):
    weights = []
    for i in range(0, len(number_of_samples_for_each_class)):
        x = assign_weight(number_of_all_samples, len(
            number_of_samples_for_each_class), number_of_samples_for_each_class[i])
        weights.append(x)
    return weights

# Delete list of indices


def deleteBasedOnIndicesList(org_lst_x, org_lst_y):
    output_x = org_lst_x.copy()
    output_y = org_lst_y.copy()
    numberOfSamplesToTruncate = max(
        count_number_of_samples_for_each_class(output_y))-2000
    i = 0
    # for i in range (0,len(output_y)-1):
    while (i < len(output_y)):
        if(numberOfSamplesToTruncate <= 0):
            break
        if (output_y[i] ==5.0):
            del output_x[i]
            del output_y[i]
            i-=1
            numberOfSamplesToTruncate-=1
        i+=1
    return output_x, output_y


def Convert(string):
    li = list(string.split(" "))
    return li

def listToString(s):
    str1 = ""
    for ele in s:
        str1 = str1 + str(ele) + " "
    return str1


def Sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2


def perf_measure(y_actual, y_hat, rate):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == rate:
            TP += 1
        if y_hat[i] == rate and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] != rate:
            TN += 1
        if y_hat[i] != rate and y_actual[i] != y_hat[i]:
            FN += 1
    return(TP, FP, TN, FN)


def calculate_metrics(TP, FP, TN, FN):
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    if (TP+FP != 0):
        precision = TP/(TP+FP)
    else:
        precision = TP
    NPV = TN/(TN+FN)
    return sensitivity, specificity, precision, NPV
