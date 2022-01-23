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


def load_data_from_CSV(filename):
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


def count_number_of_samples_for_each_class(yall, number_of_classes):
    number_of_samples = []    
    for i in range(0, number_of_classes):
        number_of_samples.append(yall.count(i))
        print(yall.count(i))
    return number_of_samples


def assign_weight(number_of_all_samples, number_of_classes, number_of_samples_for_each_class):
    return number_of_all_samples / (number_of_classes * number_of_samples_for_each_class)


def assign_weight_for_each_class(number_of_all_samples, number_of_samples_for_each_class):
    weights = []
    for i in range(0, len(number_of_samples_for_each_class)):
        x = assign_weight(number_of_all_samples, len(number_of_samples_for_each_class), number_of_samples_for_each_class[i])
        weights.append(x)
    return weights

def deleteBasedOnIndicesList(orgListX, orgListY):
    outputX = orgListX.copy()
    outputY = orgListY.copy()
    numberOfSamplesToTruncate = max(countNumberOfSamplesForEachClass(outputY))-2000
    i = 0
    while (i < len(outputY)):
        if(numberOfSamplesToTruncate <= 0):
            break
        if (outputY[i] ==5.0):
            del outputX[i]
            del outputY[i]
            i-=1
            numberOfSamplesToTruncate-=1
        i+=1
    return outputX, outputY


def convert(string):
    li = list(string.split(" "))
    return li

def list_to_string(s):
    str1 = ""
    for ele in s:
        str1 = str1 + str(ele) + " "
    return str1


def sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2


def perf_measure(y_actual, y_hat, rate):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(yHat)):
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

#def classify_rate(rate):
    #classified_rate = np.argmax(rate)
    #rate2 = rate.copy()
    #maks = max(rate)
    #rate2 = rate2[rate2 != maks]
    #maks2 = max(rate2)
    #maksIndex = np.where(rate == maks)[0][0]
    #maksIndex2 = np.where(rate == maks2)[0][0] 
    #result = -1
    #temp = findBestRate(rate, maksIndex)
    #temp2 = findBestRate(rate, maksIndex2)
    #if (temp>temp2):
    #    result = maksIndex
    #else:
    #    result = maksIndex2
    #if (i>0 and i < len(rate)-1):
     #   temp = rate[i] + 0.5*rate[i-1] + 0.5*rate[i+1]
    #elif (i == 0):
    #    temp = rate[i] + 0.5*rate[i+1]
    #else:
    #    temp = rate[i] + 0.5*rate [i-1]
    #if (temp>maks):
    #    maks = temp
    #return result

def findBestRate(rate, maksIndex):
    if (maksIndex>0 and maksIndex<len(rate)-1):
        temp = rate[maksIndex]+0.5*rate[maksIndex-1]+0.5*rate[maksIndex+1]
    elif (maksIndex == 0):
        temp = rate[maksIndex] + 0.5*rate[maksIndex+1]
    else:
        temp = rate[maksIndex] + 0.5*rate[maksIndex-1]
    return temp

def classify_rate(rate):
    classified_rate = np.argmax(rate)
    maks = 0
    maks2 = 0
    temp = 0
    max_index = 0
    for i in range (0, len(rate)):        
        if (i>0 and i < len(rate)-1):
            temp = rate[i] + 0.5*rate[i-1] + 0.5*rate[i+1]
        elif (i == 0):
            temp = rate[i] + 0.5*rate[i+1]
        else:
            temp = rate[i] + 0.5*rate [i-1]
        if (temp>maks):
            maks = temp
            max_index = i
    return max_index
