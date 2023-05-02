# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 12:45:59 2016

@author: USER
"""
#%%
from __future__ import division
import nltk, re, pprint
import pickle
#%%
text = nltk.word_tokenize("""I am Jesse Dirisu, I am from Nigeria, I have a B.A in \
Linguistics. I am presently learning NLTK""")
tags = nltk.pos_tag(text)
tags
#%%
tex = nltk.word_tokenize("They refuse to permit us to obtain the refuse permit")
nltk.pos_tag(tex)
#%%
#manually generating tag sets with nltk
#the str2tuple method comes in handy
sent = "Enini/NN keme/PRP ghi/VBP Jesukhogie/NNP"
sent_tag = [nltk.str2tuple(i) for i in nltk.word_tokenize(sent)]
sent_tag
#%%
#reading tagged corpora
from nltk.corpus import brown
set_ = brown.tagged_words()
#%%
#making pos tags more readable
#the simplify_tags = True parameter serves this purspose but is since deprciated
#it is replaced by the tagset=universal parameter
simp = nltk.pos_tag(text, tagset='universal')
simp
#%%#nltk.app.concordance()
brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')
tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)
#there is a new most_common method for cfreqdist objects
tag_fd.plot()
#%%
pos = nltk.ConditionalFreqDist(set_)
#pos['yield'].most_common()
cfd = nltk.ConditionalFreqDist((tags, words) for (words, tags) in set_)
cfd['VB'].most_common()
#cfd.conditions()
#%%
k = set_.index(('kick', 'VB'))
set_[k-1:k+1]
#%%
for i in set_:
    if i[1] == 'VB':
        j = set_.index(i)
        print(set(set_[j-1:j+1]))
#%%
#dictionaries
#python has a defaualtdict type
#using default dict eliminates key errors as non existent keys are automatically assinged default values
from collections import defaultdict

dic = defaultdict(list, {'Hoe': [1,2,3,4,5]})
#%%
#customising defaultdict types
tag_dict = defaultdict(lambda: 'NOUN', {'Kill': 'Verb'})
#%%
#tagging untagged texts
#getting the most frequent tag
maxt = [tags for word,tags in set_] 
nltk.FreqDist(maxt).max()
#%%
#mapping specifictags to all types with DefaultTagger
raw = "I do not like green eggs and ham, I do not like them Sam I am!"
tokens = nltk.word_tokenize(raw)
default_tagger = nltk.DefaultTagger('NN')
default_tagger.tag(tokens)
#%%
#tagging with the regex tagger
patterns = [
(r'\b[A-Z][a-z]+\b', 'NNP'), #proper nouns 
(r'[\)\'\"\.\,\*\+<=>]', 'SYM'), #Symbols 
(r'.*ing$' , 'VBG' ), # gerunds
(r'.*ly$', 'JJ'), # adjectives
(r'\b(W|w)h.{1,3}', 'WP'), #Wh pronoun
(r'.*ed$' , 'VBD' ), # simple past
(r'.*es$' , 'VBZ' ), # 3rd singular present
(r'.*ould$' , 'MD' ), # modals
(r'.*\'s$' , 'NN$' ), # possessive nouns
(r'.{2,}s$' , 'NNS' ), # plural nouns
(r'^-?[0-9]+(.[0-9]+)?$' , 'CD' ), # cardinal numbers
(r'.*' , 'NN' ), # nouns (default)
]

tagee = 'what is your name'
k = nltk.word_tokenize(tagee)

regtagger = nltk.RegexpTagger(patterns)
regtagger.tag(k)
#%%
sent = brown.tagged_sents(categories='news')[3]
f = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
mfw = f.most_common(100)
likely_tags = dict((word, cfd[word].max()) for (word, _) in mfw)
baseline_tagger = nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))
#baseline_tagger.evaluate(brown.tagged_sents(categories='news'))
baseline_tagger.tag(sent)
#%%
#unigram tagger
from nltk.corpus import brown
from nltk.corpus import udhr

pract = udhr.sents('English-Latin1')

tagged_sents = brown.tagged_sents(categories='news')
sents = brown.sents(categories='religion')
sent_tagger = nltk.UnigramTagger(tagged_sents)
#sent_tagger.tag(pract[1])
sent_tagger.evaluate(brown.tagged_sents(categories='religion'))
#%%
#training and testing sets
size = int(len(tagged_sents) * 0.9)
trainer = tagged_sents[:size]
tester = tagged_sents[size:]
trained_tagger = nltk.UnigramTagger(trainer)
trained_tagger.evaluate(tester)*100
#%%
#bigram taggers
#in contrasts to unigram taggers which consider only the token to be tagged in pos tagging
#the bigram tagger goes beyond this to conider the context particularly the nature of the word immediately preceeding the tag to
#ascribe a tag to the token
trained_bitagger = nltk.BigramTagger(trainer)
#bigram taggers work fine with known datasets but rather poor with unknown datasets
#the nltk.TrigramTagger would even perform worse
#for example
print(trained_bitagger.tag(brown.sents(categories='news')[2007]))
print()
print(trained_bitagger.tag(brown.sents(categories='news')[4203]))
#%%
#combining taggers
#due to the highly context sensitive nature of bigram taggers it fails to perform suitably in new contexts where novel factors prevail
#this is known in in the NLP community as the sparse data tradeoff
#to increase both accuracy and generalisability
default = nltk.DefaultTagger('NN')
fallback = nltk.UnigramTagger(trainer, backoff=default)
main = nltk.BigramTagger(trainer, backoff=fallback)
ult = nltk.TrigramTagger(trainer, backoff=main) #appart from the backoff attribute a cutoff int attribute can be specified for the Tagger object to specify the number of instances of feature to be considered when tagging
ult.evaluate(tester)
print(default.evaluate(brown.tagged_sents(categories='religion')))
#%%
#tagging unknown words
#storing trained data
import pickle
tagset = brown.tagged_sents()
foo = nltk.RegexpTagger(patterns)
uni = nltk.UnigramTagger(tagset, backoff=foo)
bi = nltk.BigramTagger(tagset, backoff=uni)
tri = nltk.TrigramTagger(tagset, backoff=bi)

#tri.evaluate(brown.tagged_sents(categories='religion'))
with open('tagset.pickle', 'wb') as f:
    pickle.dump(tri, f)
    f.close()
#%%
f = open('tagset.pickle', 'rb')
tagger = pickle.load(f)

k = nltk.word_tokenize("What is your name")
tagger.tag(k)
#%%
#testing the performance of our taggger
test_tags = [tag for sent in brown.sents() for (word,tag) in tagger.tag(sent)]
gold_tags = [tag for (word,tag) in brown.tagged_words()]
print(nltk.ConfusionMatrix(test_tags, gold_tags))
#%%
#brill or transformation based tagging
#they are more efficient than n-gram taggers as they can derive accurate analysis from rather small data dets in comparison to n-gram taggers
#brill taggers continually learn from the context to generate progresivelly more accurate taggings
uni = nltk.UnigramTagger(tagset)
tok = "The President said he will ask Congress to increase grants to states for vocational rehabilitation"
k = uni.tag(nltk.word_tokenize('be is am been being are'))
j = tagger.tag(nltk.word_tokenize('be is am been being are'))

print(k,"\n", j)

#%%
nltk.tag.brill.demo