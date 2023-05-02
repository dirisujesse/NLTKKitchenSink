# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:12:16 2017

@author: USER
"""
#%%
from __future__ import division
import re
import random
import pickle

import nltk

from nltk.corpus import names
from nltk.classify import apply_features
from nltk.corpus import movie_reviews
from nltk.corpus import brown
#%%
#classifying text
def gender_features(word):
    """This function returns a dictionary from a string argument of the last 2 characters of the string
       gender_features(string) ==> dict
    """
    return {"suffix1": word[-1:], "suffix2": word[-2:]}

gender_features("Shrek")
#%%
#creating a labelled list of features for names

labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)
#%%
#creating the feature set
feature_sets = [(gender_features(name), gender) for name, gender in labeled_names]
training_set, testing_set = feature_sets[500:], feature_sets[:500]
g_classifier = nltk.NaiveBayesClassifier.train(training_set)
#%%
#testing the accuracy of a clasifier
print(nltk.classify.accuracy(g_classifier, testing_set) * 100, "%")
g_classifier.show_most_informative_features()
#%%
#applying features on the go
training_set = apply_features(gender_features, labeled_names[500:])
testing_set = apply_features(gender_features, labeled_names[:500])
n_classifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(n_classifier, testing_set) * 100, "%")
n_classifier.show_most_informative_features()
#%%
def gender_features2(name):
    """This function returns a dictionary from a string argument of the last 2 characters of the string
       gender_features(string) ==> dict
    """
    features = {}
    features["first_letter"] = name[0].lower()
    features["last_letter"] = name[-1].lower()
    for letter in "abcdefghijklmnopqrstuvwxyz":
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features
#%%
#development sets
train = labeled_names[1500:]
dev_names = labeled_names[200:1200]
test = labeled_names[:500]
#%%
#using the dev set
train_set = [(gender_features(n), gender) for n, gender in train]
dev_set = [(gender_features(n), gender) for n, gender in dev_names]
test_set = [(gender_features(n), gender) for n, gender in test]

classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, dev_set) * 100,'%')
classifier.show_most_informative_features()
#%%
#intercepting errors with dev sets
errors = []
for (name, tag) in dev_names:
    guess = classifier.classify(gender_features(name))
    if guess != tag:
        errors.append((tag, guess, name))
print(len(errors))

for (tag, guess, name) in sorted(errors):
    print('correct={:<8} guess={:<8s} name={:<30}'.format(tag, guess, name))
#%%
#document classification
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)
#%%
#constructing a feature extractor
##from nltk.corpus import stopwords
import nltk

from nltk.corpus import stopwords
eng = stopwords.words("english")
eng

all_words = nltk.FreqDist([w.lower() for w in set(movie_reviews.words()) if w not in eng])
word_features = list(all_words)[:2000]

def document_features(document):
    features = {}
    for word in word_features:
        features["contains({})".format(word)] = (word in set(document))
    return features
print(document_features(movie_reviews.words('pos/cv957_8737.txt')))
#%%
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set =  featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features()
#%%
sfdist = nltk.FreqDist()
for word in brown.words():
    sfdist[word[-1:]] += 1
    sfdist[word[-2:]] += 1
    sfdist[word[-3:]] += 1

common_suffixes = [(suffix) for suffix, count in sfdist.most_common(100)]
common_suffixes
#%%
def pos_features(word):
    features = {}
    for suffix in common_suffixes:
        features["endswith({})".format(suffix)] = word.lower().endswith(suffix)
    return features

feature_sets = [(pos_features(w), tag) for w, tag in brown.tagged_words(categories='news')]

size = int(len(feature_sets) * 0.3)

train_set, test_set = feature_sets[size:], feature_sets[:size]

classifier = nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
print(classifier.pseudocode(depth=4))
#%%
k = brown.sents()
tagged = brown.tagged_sents(categories='news')

def loc_features(const, i):
    features = {'suffix(1)': const[i][-1:], 'suffix(2)': const[i][-2:],
    'suffix(3)': const[i][-3:]}

    if i == 0:
        features["Prev_Word"] = "<START>"
    else:
        features["Prev_Word"] = const[i-1]
        
    return features

#loc_features(k[0], 8)

featureset = []

for i in tagged:
    untagged = nltk.tag.untag(i)
    for j, (word, tag) in enumerate(i):
        featureset.append((loc_features(untagged, j), tag))

size = int(len(featureset) * 0.1)
train_set, test_set = featureset[size:], featureset[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)
#%%
#sentence segmentation
sents = nltk.corpus.treebank_raw.sents()
tokens = []
boundaries = set()
offset = 0
for sent in sents:
    tokens.extend(sent)
    offset += len(sent)
    boundaries.add(offset-1)
#%%
def punct_features(tokens, i):
    return {'next-word-capitalised': tokens[i+1][0].isupper(),
            'prev-word': tokens[i-1].lower(),
            'punct': tokens[i],
            'prev-word-is-one-char': len(tokens[i-1]) == 1}
#%%
featuresets = [(punct_features(tokens, i), (i in boundaries))
                for i in range(1, len(tokens)-1)
                if tokens[i] in '.?!']
#%%
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier, test_set)

#def segment_sentences(words):
#    start = 0
#    sents = []
#    for i, word in enumerate(words):
#        if word in '.?!' and classifier.classify(punct_features(words, i)) == True:
#            sents.append(words[start:i+1])
#            start = i+1
#    if start < len(words):
#        sents.append(words[start:])
#    return sents
#%%
#capturing speech acts
posts = nltk.corpus.nps_chat.xml_posts()[:10000]
def act_feat(post):
    for word in nltk.word_tokenize(post):
        return {'contains({})'.format(word.lower()): True}
                
fs = [(act_feat(post.text), post.get('class'))
      for post in posts]
size = int(len(fs) * 0.4)
train, test = fs[size:], fs[:size]
classer = nltk.NaiveBayesClassifier.train(train)
nltk.classify.accuracy(classer, test)
#%%
rt = nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
ext = nltk.RTEFeatureExtractor(rt)
print(ext.hyp_words)
#%%
#Entropy and Info Gain algorithm for decision tree classification
import numpy as np
def entropy(labels):
    freq = nltk.FreqDist(labels)
    prob = [freq.freq(l) for l in freq]
    return "{:.3f}".format(abs(-sum(p * np.log2(p) for p in prob)))
    