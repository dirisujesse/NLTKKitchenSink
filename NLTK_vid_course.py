 # -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:47:00 2016

@author: Jesse Dirisu
"""
#%%
from nltk.tokenize import sent_tokenize, word_tokenize
#%%
"""Video 1 basic tokenisation"""
fir = "Jesse is 21 years old, and Kiki is 17. Emma is 73 years old, while his grand father Ona is 121 years old."
sent1 = "Hello Babe, how you dey? The weather dey nice and python dey cool. The sky is yellow e be like sun don blow."
#==============================================================================
# print(sent_tokenize(sent1))
# print(word_tokenize(sent1))
# 
# for i in word_tokenize(sent1):
#     print(i)
# for y in sent_tokenize(sent1):
#     print(y)
# I will now attempt building a tokenizing function
#==============================================================================
def wordtokenizer(x):
    """this tokenizes words in texts"""
    from nltk.tokenize import word_tokenize
    print(word_tokenize(x))
    
    for i in word_tokenize(x):
        print(i)

def senttokenizer(x):
    """this tokenizes sentences in texts"""
    from nltk.tokenize import sent_tokenize
    print(sent_tokenize(x))
    
    for i in sent_tokenize(x):
        print(i, end="\n")
#%%
"""Video 2 stop words"""
#==============================================================================
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# sent = "I am feeling so good today that i feel like taking you all out for a treat"
# stopwordz = set(stopwords.words('english'))
# 
# word = word_tokenize(sent)
# 
# a = []
# 
# for words in word:
#     if words not in stopwordz:
#         a.append(words)
#         
# print(a)
#==============================================================================
sent = "I am feeling so good today that i feel like taking you all out for a treat"
def filter(x):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    stopwordz = set(stopwords.words('english'))
    word = word_tokenize(x)

    a = []
    for words in word:
        if words not in stopwordz:
            a.append(words)
            
    print(a) 
#stopwords.raw('english')
#stopwords.word('english')
#%%
#stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()
#words = ["Eat","Eating","Eats","Sing","Singing","Sings","Run","Running","Runs","Cry","Crying"]
#for w in words:
#    print(ps.stem(w))
#    print(ps.stem_word(w))
#    print(ps.vowels(w))

sent = " Jesse said that he will eat the goat, while he was eating the cow leg, he eats too much."
sen = "Sing he said to me while singing, that lovely song, which he sings a lot"
se = "Run for your life my dear, as you can see I am running, he who runs lives to fight another day, now is not the time to cry, quit crying!"

def words(x):
    y = word_tokenize(x)
    for i in y:
        print(ps.stem(i))
        
        
#%%
#part of speech tagging
#==============================================================================
 import nltk
 from nltk.corpus import state_union
 from nltk.tokenize import PunktSentenceTokenizer
 
 i = state_union.raw("2005-GWBush.txt")
 j = state_union.raw("2006-GWBush.txt")
 
 x = PunktSentenceTokenizer(i)
 y = x.tokenize(j)
 
 def processor():
     try:
         for i in y:
             words = nltk.word_tokenize(i)
             tags = nltk.pos_tag(words)
             print(tags)
         
     except Exception as e:
         print(str(e))
 
 processor()
#==============================================================================

#==============================================================================
# def tagger(x):
#     import nltk
#     from  nltk.corpus import x
#     from nltk.tokenize import PunktSentenceTokenizer
#     j = input("enter the first fileid: ")
#     k = input("enter the next fileid: ")
#     
#     i = x.raw("j")
#     l = x.raw("k")
#     
#     q = PunktSentenceTokenizer(i)
#     y = q.tokenize(l)
#     
#     try:
#         for word in y:
#             e = nltk.word_tokenize(word)
#             f = nltk.pos_tag(e)
#             print(e)
#         
#     except Exception as e:
#         print(str(e))
#==============================================================================
#%%
#Chunking
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
i = state_union.raw("2005-GWBush.txt")
j = state_union.raw("2006-GWBush.txt")
 
x = PunktSentenceTokenizer(i)
y = x.tokenize(j)
 
def chunker():
    try:
        for i in y:
            words = nltk.word_tokenize(i)
            tags = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tags)
            chunked.draw()
        
    except Exception as e:
        print(str(e))

#%%
#chinking
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
i = state_union.raw("2005-GWBush.txt")
j = state_union.raw("2006-GWBush.txt")
 
x = PunktSentenceTokenizer(i)
y = x.tokenize(j)
 
def chinker():
    try:
        for i in y:
            words = nltk.word_tokenize(i)
            tags = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<.*>+}
                        }<VB.?|DT|IN>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tags)
            print(chunked)
            chunked.draw()
        
    except Exception as e:
        print(str(e))
#%%
#named entity recognition
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 
i = state_union.raw("2005-GWBush.txt")
j = state_union.raw("2006-GWBush.txt")
 
x = PunktSentenceTokenizer(i)
y = x.tokenize(j)

def namident():
    try:
        for i in y[:2]:
            words = nltk.word_tokenize(i)
            tags = nltk.pos_tag(words)
            #you can place binary = True statements inthe parenthese to eliminate entity types
            nament = nltk.ne_chunk(tags, binary=True)
            
            nament.draw()
            
    except Exception as e:
        print(str(e))
#%%
#Lemmatizing
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats", pos="n"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
#%%
#.lemmas method works on individual sysets
#working with wordnet
from nltk.corpus import wordnet

syn = wordnet.synsets("program")
print(syn)

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
#    synonyms.append(syn.lemma_names())
#    #anonyms.append(syn.antonyms())
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
    
print(set(synonyms))
print(set(antonyms))

#testing for semantic simmilarity
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")



x = w1.wup_similarity(w2) #returns a float which if multiplied by 100 is the jugded percentage of similarity between the words

perc = x * 100

print("The level of similarity between both words is {}%".format(int(perc)))
#%%
def word_syn_checker(x, y):
    w, z = wordnet.synsets(x)[0], wordnet.synsets(y)[0]
    
    j = w.wup_similarity(z)
    
    perc = j * 100
    print("The level of similarity between both words is {}%".format(int(perc)))
#%%
#text classification for sentiment analysis
import nltk
import random
from nltk.corpus import movie_reviews, stopwords

stop_words = stopwords.words('english')

documents = [(list(movie_reviews.words(fileid)), category)
for category in movie_reviews.categories()
for fileid in movie_reviews.fileids(category)
]

random.shuffle(documents)

w_list = []
for w in movie_reviews.words():
    w_list.append(w.lower())

a_list = nltk.FreqDist([w for w in w_list if w.isalpha() and w not in stop_words])
print(a_list.most_common(15))
#%%
#text classification for sentiment analysis
import nltk
import random
from nltk.corpus import movie_reviews

word_features = list(a_list.keys())[:3000]
print(word_features[:200])

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features
    
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featureset = [(find_features(rev), category) for rev, category in documents]
#%%
#naive bayes algorithims
import random

training_set = featureset[:1900]
testing_set = featureset[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive bayes accuracy percentage {}%'.format(nltk.classify.accuracy(classifier, testing_set)*100))
classifier.show_most_informative_features(15)
#%%
#pickling
import pickle
f = open('classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()
#%%
#sk learn classifier
from nltk.classify.scikitlearn import SklearnClassifier as Sk
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

MNB_group = Sk(MultinomialNB())
MNB_group.train(training_set)
print('MNB accuracy percentage {}%'.format(nltk.classify.accuracy(MNB_group, testing_set)*100))

#GNB_group = Sk(GaussianNB())
#GNB_group.train(training_set)
#print('GNB accuracy percentage {}%'.format(nltk.classify.accuracy(GNB_group, testing_set)*100))

BNB_group = Sk(BernoulliNB())
BNB_group.train(training_set)
print('BNB accuracy percentage {}%'.format(nltk.classify.accuracy(BNB_group, testing_set)*100))

#SVC, LinearSVC, NuSVC

LogisticRegression_group = Sk(LogisticRegression())
LogisticRegression_group.train(training_set)
print('LNB accuracy percentage {}%'.format(nltk.classify.accuracy
      (LogisticRegression_group, testing_set)*100))

SGDClassifier_group = Sk(SGDClassifier())
SGDClassifier_group.train(training_set)
print('SNB accuracy percentage {}%'.format(nltk.classify.accuracy
      (SGDClassifier_group, testing_set)*100))

SVC_group = Sk(SVC())
SVC_group.train(training_set)
print('SVNB accuracy percentage {}%'.format(nltk.classify.accuracy
      (SVC_group, testing_set)*100))

LinearSVC_group = Sk(LinearSVC())
LinearSVC_group.train(training_set)
print('LSNB accuracy percentage {}%'.format(nltk.classify.accuracy
      (LinearSVC_group, testing_set)*100))

NuSVC_group = Sk(NuSVC())
NuSVC_group.train(training_set)
print('NUNB accuracy percentage {}%'.format(nltk.classify.accuracy
      (NuSVC_group, testing_set)*100))
