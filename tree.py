
# coding: utf-8

# # Syntax with NLTK

# In[1]:

import nltk
import re
import pprint
# * ** PSG based analysis **

# In[2]:

uneme_grammar = nltk.CFG.fromstring("""
S -> NP VP | NP VP COMP
PP -> P NP
NP -> Det N | N Det | N | NP PP| PRP | N PRP PRP
VP -> V | V NP | V PP | V-bar COMP | V COMP
V-bar -> V NP | V PP
COMP -> WDT S
N -> 'owa' | 'omo' | 'kpiro' | 'emale' | 'onu' | 'Alizhi' | 'Onashiki' | 'Eko'
V -> 'le' | 'ri' | 'gbe' | 'du' | 'sene' | 'guma' | 'vwe' 
P -> 'ekeli' | 'i' | 'zhi'| 'efe' | 'ki' | 'weli'
PRP -> 'mi' | 'u' | 'or' | 'mwan'| 'wa'| 'eh' | 'me' | 'er' | 'a' | 'wowe' | 'ele' | 'ole'
WDT -> 'ni'
Det -> 'oni' | 'eni' | 'okpa' | 'eva' | 'keme' | 'ke' | 'na' | 'nhi'
""")


# In[3]:

sent = ['omo', 'na', 'guma', 'me', 'ni', 'u', 'vwe', 'onu', 'ke']


# In[11]:

parser = nltk.ChartParser(uneme_grammar)


# In[50]:

for tree in parser.parse(sent):
    tree.draw()
    print(tree)


# In[4]:

sent2 = "Onashiki ole or gbe Alizhi".split(" ")
sent3 = "Alizhi le emale weli kpiro keme i Eko".split(" ")
print(sent2,sent3)


# In[12]:

for tree in parser.parse(sent2):
    print(tree)
    tree.draw()


# In[ ]:

parser2 = nltk.RecursiveDescentParser(uneme_grammar)


# In[13]:

for tree in parser2.parse(sent3):
    print(tree)
    tree.draw()


# In[ ]:

nltk.data.load('file:mygrammar.cfg')


# In[ ]:



