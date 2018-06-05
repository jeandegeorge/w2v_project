#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:39:47 2018

@author: philippehayat
"""

from __future__ import division
import argparse
import pandas as pd
from string import punctuation
import pickle

# useful stuff
import numpy as np
from numpy.random import choice
from scipy.special import expit
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import csv


__authors__ = ['Jean Degeorge', 'Kilian Tep', 'Paul Hayat']
__emails__ = ['jean.degeorge@essec.edu', 'kilian.tep@essec.edu', 'paul.hayat@essec.edu']

#path = '/Users/kiliantep/NLP1.txt'


# function that transforms text into list of sentences
def text2sentences(path):
    sentences = []
    with open(path) as f:
        for l in f:
            sentences.append(l.lower().split('.'))

    sentences = sum(sentences, []) 

    sentences = [sentences[j] for j in range(len(sentences))]

    for j in range(len(sentences)):
        sentences[j] = sentences[j].split()
    a = '\n'
    punctuation_plus = punctuation + a
    for j in range(len(sentences)):
        if sentences[j] != a:
            sentences[j] = [''.join(c for c in s if c not in punctuation_plus) for s in sentences[j]]
            sentences[j] = [s for s in sentences[j] if s]
    sentences = [sentence for sentence in sentences if sentence != []]
        
    return sentences
#sentences = text2sentences(path)

# generate word pairs with window size = 2. 
def loadPairs(sentences):
    window_size = 2

    # generate data set of word context pairs
    pairs = []
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - window_size, 0): min(word_index + window_size, len(sentence)) + 1]:
                if nb_word != word:
                    pairs.append([word, nb_word])

    # remove duplicates
    pairs = list(map(list, pairs))

    return pairs

#data = loadPairs(path)

# sigmoid
def sigmoid(x):
    if x > 6:
        return 1.0
    elif x < -6:
        return 0.0
    else:
        return 1 / (1 + np.exp(-x))

# define derivative functions
def word_derivative( x, y, z):
    batch_index = []
    gradient = -(1-sigmoid(np.dot(x, y)))*y
    for i in batch_index:
        gradient += np.dot((1-sigmoid(-np.dot(x, z[i]))), z[i])
    return gradient

def pos_derivative(x, y, z):
    gradient = -(1-sigmoid(np.dot(x, y)))*x
    return gradient

def neg_derivative(x, y, z, indices):
    gradient = []
    for i in indices:
        gradient.append((1-sigmoid(np.dot(x, z[i])))*x)
    return gradient


class mSkipGram:
    def __init__(self, sentences, nEmbed=100, neg_size=5, window_size = 2, minCount = 5):
        
        # word count
        self.dict_freq = {}

        #loads text and removes punctuation
        self.sentences = sentences

        self.sentences_flat = sum(self.sentences, [])
        
        self.word_count = 0


        #building frequency dictionary
        for i in self.sentences_flat:
            self.word_count = self.sentences_flat.count(i)  # Python's count function, count()
            self.dict_freq[i] = self.word_count

        #creates dataset
        self.data = loadPairs(self.sentences)

        self.words = sum(self.sentences, [])
        self.words = sorted(set(self.words), key=self.words.index) # so that all duplicate words are removed
        self.word2int = {}
        self.int2word = {}
        self.vocab_size = len(self.words)  # gives the total number of unique words

        for i, word in enumerate(self.words):
            self.word2int[word] = i
            self.int2word[i] = word

        #define window size
        self.window_size = 2

        #convert vocab to list.
        self.elements = sorted(list(self.words))

        # create random vectors
        self.wordvecs = []
        for i in self.words:
            self.wordvecs.append(np.random.random(self.vocab_size))

        # define negative sample size
        self.neg_size = 5

    
    def update(self, x, y,data_word):
    
        x.append(self.wordvecs[self.word2int[data_word[0]]])
        y.append(self.wordvecs[self.word2int[data_word[1]]])


    def train(self,array_of_words,epochs=5):
        
        data = loadPairs(array_of_words)

        for epoch in range(epochs):
            print("epoch",(epoch+1),"starting")
        
            # Generate training set with word vectors instead of strings:
            X_train = [] # input word
            y_train = [] # output word

            for data_word in data:
                #print(data_word)
                self.update(X_train, y_train,data_word)

            print("epoch",(epoch+1),"data update complete")
            # training loop len(data):
            # For each unique pair:
            for j in range(len(data)):

                # power law weights & creating batch
                self.weights = []
                batch = []

                # for each unique word:
                # Recall: dict_freq is a dictionary containing the number of times each unique word appears.
                for i in range(len(self.elements)):
                    self.weights.append(self.dict_freq[self.elements[i]]) # we give to list 'weights' the frequence of each word.

                # elements = list of unique words.
                # get elements that are not in context of target word and that are not the target word itself:
                neg_indices = []
                context_indices = [i for i, x in enumerate([data[k][0] for k in range(len(data))]) if x == data[j][0]]
                context_words = [data[i][1] for i in context_indices]

                for i in range(len(data)):
                    if (data[i][0] != data[j][0]) & (data[i][1] not in context_words) & (data[i][1] != data[j][0]):
                        neg_indices.append(i)

                self.neg_elements = []
                self.neg_elements = [data[i][1] for i in neg_indices]
                self.neg_elements = sorted(set(self.neg_elements), key=self.neg_elements.index)

                if len(self.neg_elements) >= self.neg_size:

                    # create distribution
                    self.weights = np.power(self.weights, 0.75)

                    self.vocab_index = []
                    for i in range(len(self.neg_elements)):
                        self.vocab_index.append(self.word2int[self.neg_elements[i]])

                    self.weights = list(self.weights[i] for i in self.vocab_index)

                    self.weights = [i / sum(self.weights) for i in self.weights]

                    temp = list(choice(self.neg_elements, self.neg_size, p=self.weights, replace=False))

                    #while not(set(temp).issubset(neg_elements)):
                    #temp = list(choice(elements, neg_size, p=weights, replace=False))

                    batch.append(temp)

                    batch = batch[0] # flatten list

                    del temp

                    # generate batch_index: list of indices for words in batch
                    batch_index = []

                    for i in range(self.neg_size):
                        temp = self.word2int[batch[i]]
                        batch_index.append(temp)
                        del temp

                else:
                    batch_index = []

                # objective function
                J = 0
                for i in range(len(data)): # not vocab size
                    b = sigmoid(np.dot(X_train[i], y_train[i]))
                    if b != 0:
                        J += np.log(b)
                for k in batch_index:
                    a = sigmoid(np.dot(X_train[i], y_train[k]))
                    if a != 0:
                        J += np.log(a)
                J = -J

                # Stochastic gradient descent
                # w = self.word2int[data[j][0]]
                # c = self.word2int[data[j][1]]

                w = self.word2int[data[j][0]]
                c = self.word2int[data[j][1]]

                self.wordvecs[w] = self.wordvecs[w] - word_derivative(self.wordvecs[w], y_train[j], self.wordvecs)

                self.wordvecs[c] = self.wordvecs[c] - pos_derivative(self.wordvecs[w], y_train[j], self.wordvecs)

                for i in batch_index:
                    self.wordvecs[i] = self.wordvecs[i] - neg_derivative(self.wordvecs[c], y_train[j], self.wordvecs, batch_index)[k in range(self.neg_size)]


                del c
                del w

                if j == int(len(data)/4):
                    print("25 percent of epoch training completed")

                if j == int(len(data)/2):
                    print("50 percent of epoch training completed")

                if j == int(len(data)/(4/3)):
                    print("75 percent of epoch training completed")

                if j == int(len(data)-1):
                    print("100 percent of epoch training completed")
                #temp = temp[0]

                #wordvecs = np.asarray(wordvecs)
                #for (i, k) in batch_index:
                    #wordvecs[i] = temp[int2word[i]]


                #wordvecs = wordvecs.tolist()
                #del temp

            print("epoch",(epoch+1),"ending")
            print("")

        #dictionary: word as key, vector as value    
        self.vocab_dict = dict(zip(self.words, self.wordvecs))
        print (self.vocab_dict.keys())

    def save(self,filename):
        pickle.dump(self, open(filename, "wb") )
        
    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        
        # cosine similarity: Dot(word1, word2)/L2norm(word1)*L2norm(word2).
        # we calculate the cosine similarity between two words. 
        # print(cosine_similarity(X_train, y_train))
        weights1 = self.vocab_dict[word1]

        weights2 = self.vocab_dict[word2]

        cos_sim = np.dot(weights1, weights2)/(np.linalg.norm(weights1, 2)*np.linalg.norm(weights2, 2))

        return cos_sim
        #print(cosine_similarity(self.wordvecs[word1],self.wordvecs[word2]))

    @staticmethod
    def load(filename):
        return pickle.load(open(filename,'rb'))
        
# main. 
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')


    opts = parser.parse_args()

    if not opts.test:
        print('start')
        sentences = text2sentences(opts.text)
        sg = mSkipGram(sentences)
        sg.train(sentences)
        sg.save(opts.model)

    else:
        test_text = text2sentences(opts.text)
        pairs = loadPairs(test_text)

        sg = mSkipGram.load(opts.model)
        for a,b in pairs:
            try:
                print (a,b,sg.similarity(a,b))
            except KeyError:
                print(a,"or",b,"is not present in training vocab")
                continue
        print('queen','king', sg.similarity('queen','king'))

#########################################################
"""
#if __name__ == '__main__':
    # parser.add_argument('--text', help='path containing training data', required=True)
    # parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    # parser.add_argument('--test', help='enters test mode', action='store_true')

    # opts = parser.parse_args()

    # if not opts.test:
    """
"""
def main():
    print('start')
    sentences = text2sentences(path)
    sg = mySkipGram(sentences)
    #sg.train(0, 1)
    print(sg.similarity('the','call'))
    print('end')
#sg.save(opts.model)

# else:
    data = loadPairs(path)


# sg = mySkipGram.load(opts.model)
    for a, b in data:
        try:
            print(a, b, sg.similarity(a, b));
        except:
            pass

main() 
"""
#########################################################
"""
with open('pairs.csv', 'w')  as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['word1','word2', 'similarity']);
        for a, b in pairs:
            try:
                filewriter.writerow([a, b, sg.similarity(a, b)]);
            except:
                pass
""" 
