# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:01:28 2016

@author: USER
"""
#%%
"""
text4.concordance('academia')
text2.similar('hate')
text9.common_contexts(['men','women'])
FreqDist(text8).hapaxes()
text.count(x)
text.collocation(n-n)
FreqDist(text8)['God']
FreqDist(text3).keys()
FreqDist(text3).items()
FreqDist(text3).plot(50, cumulative=True)
FreqDist(text5)
text3.dispersion_plot(["God", "created", "jesus", "Adam", "Eve", "Abraham"])

Table 1-4. Some word comparison operators
Function Meaning
s.startswith(t) Test if s starts with t
s.endswith(t) Test if s ends with t
t in s Test if t is contained inside s
s.islower() Test if all cased characters in s are lowercase
s.isupper() Test if all cased characters in s are uppercase
s.isalpha() Test if all characters in s are alphabetic
s.isalnum() Test if all characters in s are alphanumeric
s.isdigit() Test if all characters in s are digits
s.istitle() Test if s is titlecased (all words in s have initial capitals)

Opening nltk corpus
nltk.corpus.item.words() to view words
nltk.corpus.item.sents() to view sentences
nltk.corpus.item.raw() to view characters
nltk.corpus.item.fileids() to view list of the names of files in the item
item = nltk.Text(nltk.corpus.corpustype.words('fileid'))
item.func(arguments)
from nltk.corpus import gutenberg
fro nltk.corpus import webtext
from nltk.corpus import nps_chat this has a post method
from nltk.corpus import brown has a category method
from nltk.corpus import reuters
from nltk.corpus import inaugural
from nltk.corpus import toolbox
nltk.corpus.names

fileids() The files of the corpus
fileids([categories]) The files of the corpus corresponding to these categories
categories() The categories of the corpus
categories([fileids]) The categories of the corpus corresponding to these files
raw() The raw content of the corpus
raw(fileids=[f1,f2,f3]) The raw content of the specified files
raw(categories=[c1,c2]) The raw content of the specified categories
words() The words of the whole corpus
words(fileids=[f1,f2,f3]) The words of the specified fileids
words(categories=[c1,c2]) The words of the specified categories
sents() The sentences of the specified categories
sents(fileids=[f1,f2,f3]) The sentences of the specified fileids
sents(categories=[c1,c2]) The sentences of the specified categories
abspath(fileid) The location of the given file on disk
encoding(fileid) The encoding of the file (if known)
open(fileid) Open a stream for reading the given corpus file
root() The path to the root of locally installed corpus
readme() The contents of the README file of the corpus


loading your corpus
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
['README', 'connectives', 'propernames', 'web2', 'web2a', 'words']
wordlists.words('connectives')
['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', ...]
or

"""
#%%
V = set(text1)
long_words = [w for w in V if len(w) > 15]
sorted(long_words)
#%%
def selective(x):
    for word in x:
        if word.endswith('st'):
            print(word)
            #print(len(list(set(word)))
#%%
def collocator(x):
    return x.collocations()
#%%
def freqgetter(x):
    freq = FreqDist([len(w) for w in x])
    list(freq.keys())
    return freq.items()
#%%
def catch(x):
    iter_ = FreqDist([len(i) for i in x])
    print (list(iter_.keys()))
    print()
    print(list(iter_.items()))
#%%
def catchdemlong(x):
    for w in x:
        if len(w) > 15:  
            print(list(w))
#%%
def howmanytimes(x):         
    for w in x:
        if len(w) >= 10:
            print(list(len(set(w))))
#%%Example Description
'''
fdist = FreqDist(samples) Create a frequency distribution containing the given samples
fdist.inc(sample) Increment the count for this sample
fdist['monstrous'] Count of the number of times a given sample occurred
fdist.freq('monstrous') Frequency of a given sample
fdist.N() Total number of samples
fdist.keys() The samples sorted in order of decreasing frequency
for sample in fdist: Iterate over the samples, in order of decreasing frequency
fdist.max() Sample with the greatest count
fdist.tabulate() Tabulate the frequency distribution
fdist.plot() Graphical plot of the frequency distribution
fdist.plot(cumulative=True) Cumulative plot of the frequency distribution
fdist1 < fdist2 Test if samples in fdist1 occur less frequently than in fdist2
'''
#%%
sorted([w for w in set(text7) if '-' in w and 'index' in w])
#%%
sorted([wd for wd in set(text3) if wd.istitle() and len(wd) > 10])
#%%
sorted([w for w in set(sent7) if not w.islower()])
#%%
sorted([t for t in set(text2) if 'cie' in t or 'cei' in t])
#%%
#%%
def cant(x):
    dmay = FreqDist([w for w in x if w.istitle() and not w.isdigit()])
    dmay.tabulate()
    dmay.plot(20, cumulative=True)
    print(list(dmay))
    print(len(set(dmay)))
    print(len(dmay))
#%%    
def cann(tex):
    dmay = FreqDist([w for w in tex if w.istitle() and not w.isdigit()])
    dmay.tabulate()
    dmay.plot(20, cumulative=True)
    print(list(set(dmay)))
    print(len(dmay))
    print(len(set(dmay)))
#%%
def iter_(x):
    longy = [w for w in x if w.isalpha and len(w) >= 15]
    print(longy)
    print('The lenght of the list of words 15 or more letters long in',x,'is',len(longy))
    print('The lenght of the list of unique words 15 or more letters long in',x,'is',len(set(longy)))
#%%    
def freq(x):
    longy = [w for w in x if w.isalpha and len(w) >= 15]
    fre = FreqDist(longy)
    fre.plot(20, cumulative=True)
    print(set(fre))
    print(fre.items())
    print(fre.keys())     
        
#%%
def counter(x):
    for fileid in x.fileids(): 
        num_chars = len(x.raw(fileid))
        num_words = len(x.words(fileid))
        num_sents = len(x.sents(fileid))
        num_vocab = len(set([w.lower() for w in x.words(fileid)]))
        print(int(num_chars), int(num_words), int(num_sents), int(num_vocab), int(num_chars/num_words), int(num_words/num_sents), int(num_words/num_vocab),
        fileid)
#%%
#def importer():
 #   x = input('Enter the title of the text you want to view: ')
  #  from nltk.corpus import x
   # x.fileids()     
from nltk.corpus import webtext
list(webtext.fileids())
for i in webtext.fileids():
    print(i, webtext.raw(i)[:100])
#%%
from nltk.corpus import brown
news_text = brown.words(categories='news')
fdist = nltk.FreqDist([w.lower() for w in news_text])
wh = ['what', 'why', 'where', 'when', 'how', 'which']
for m in wh:
    print(m + ':', fdist[m])
    
#def brownie(x,c):
 #   brown.words(categories = x)
  #  z = nltk.FreqDist([w.lower() for w in x])
   # for i in c:
    #    print(i + ':', fdist[i])
#%%
tabulator = nltk.ConditionalFreqDist( (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre))
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
tabulator.tabulate(conditions=genres, samples=modals)
#%%
def plotter(x):
    cfd = nltk.ConditionalFreqDist(
        (target, fileid[:4])
        for fileid in x.fileids()
        for w in x.words(fileid)
        for target in ['what', 'why', 'where', 'when', 'how', 'which']
        if w.lower().startswith(target))
    cfd.plot()
    #cfd.tabulate(conditions=, samples=target)
#%%
from nltk.corpus import udhr
udhr.fileids()
wey = udhr.fileids()[:10]
for i in wey:
    sey = udhr.raw(i)
    FreqDist(sey).plot(26, cumulative=True)
#%%
me = udhr.fileids()[300:306]
for i in me:
    him = udhr.raw(i)
    nltk.FreqDist(him).plot(20, cumulative=True)
#%%
languages = ['Chickasaw', 'English', 'German_Deutsch',
 'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik']
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1'))
cfd.plot(cumulative=True)
cfd.tabulate(conditions=languages, samples=range(20), cumulative=True)
#%%
nu = nltk.FreqDist([len(word) for word in udhr.words("Edo-Latin1")])
#%%
#you load your corpus with nltk PlaintextCorpusReader and some regex
from nltk.corpus import PlaintextCorpusReader
corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()

#or
#==============================================================================
# from nltk.corpus import BracketParseCorpusReader
# corpus_root = r"C:\corpora\penntreebank\parsed\mrg\wsj"
# file_pattern = r".*/wsj_.*\.mrg"
# ptb = BracketParseCorpusReader(corpus_root, file_pattern)
# ptb.fileids()
# ['00/wsj_0001.mrg', '00/wsj_0002.mrg', '00/wsj_0003.mrg', '00/wsj_0004.mrg', ...]
# len(ptb.sents())
# 49208
# ptb.sents(fileids='20/wsj_2013.mrg')[19]
# ['The', '55-year-old', 'Mr.', 'Noriega', 'is', "n't", 'as', 'smooth', 'as', 'the',
# 'shah', 'of', 'Iran', ',', 'as', 'well-born', 'as', 'Nicaragua', "'s", 'Anastasio',
# 'Somoza', ',', 'as', 'imperial', 'as', 'Ferdinand', 'Marcos', 'of', 'the', 'Philippines',
# 'or', 'as', 'bloody', 'as', 'Haiti', "'s", 'Baby', Doc', 'Duvalier', '.']
#==============================================================================
#%%
def cond(x):
    cond_pair = [(genre, word) for genre in x.categories()[:10] for word in x.words(categories=genre)]
    i = nltk.ConditionalFreqDist(cond_pair)
    i
    i.tabulate()
#%%
cfd = nltk.ConditionalFreqDist( (word, genre)
                     for genre in brown.categories()
                     for word in brown.words(categories=genre))
                         
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Satuday', 'Sunday']
cfd.tabulate(conditions=['news', 'romance'], samples=days)
#%%
#bigrams
#days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Satuday', 'Sunday']
#k = nltk.bigrams(days)
#for i in k:
#    print(list(k))
    
def generate_model(cfdist, word, num=15):
    for i in range(num):
        print(word)
        word = cfdist[word].max()
        

text = nltk.corpus.genesis.words('english-kjv.txt')
bigrams = nltk.bigrams(text)
cfe = nltk.ConditionalFreqDist(bigrams)
"""
#common conditional freq dist methods
Table 2-4. NLTK’s conditional frequency distributions: Commonly used methods and idioms for
defining, accessing, and visualizing a conditional frequency distribution of counters
Example Description
cfdist = ConditionalFreqDist(pairs) Create a conditional frequency distribution from a list of pairs
cfdist.conditions() Alphabetically sorted list of conditions
cfdist[condition] The frequency distribution for this condition
cfdist[condition][sample] Frequency for the given sample for this condition
cfdist.tabulate() Tabulate the conditional frequency distribution
cfdist.tabulate(samples, conditions) Tabulation limited to the specified samples and conditions
cfdist.plot() Graphical plot of the conditional frequency distribution
cfdist.plot(samples, conditions) Graphical plot limited to the specified samples and conditions
cfdist1 < cfdist2 Test if samples in cfdist1 occur less frequently than in cfdist2
"""
#%%
def plural(word):
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'
#%%

def rare_words(text):
    text_vocab = set(w.islower() for w in text if w.isalpha())
    enl_vocab = set(word.islower() for word in nltk.corpus.words.words(''))
    rare = text_vocab.difference(enl_vocab)
    return sorted(rare)
#%%
from nltk.corpus import names

def common():
    j = names.words('female.txt')
    x = names.words('male.txt')
    l = [z for z in j if z in x]
    print(sorted(set(l)))
    #nltk.FreqDist(set(l)).plot()
    
cfdist = nltk.ConditionalFreqDist((fileids, word[-1])
    for fileids in names.fileids()
    for word in names.words(fileids))
cfdist.plot()
#%%
#Arpabet a list of phonetic codes for computers
entries = nltk.corpus.cmudict.entries()
len(entries)
#for entry in entries[0:70]:
#    print(entry)
    
for word, pc in entries:
    if len(pc) == 3:
        pc1, pc2, pc3 = pc
        if pc1 == 'F' and pc3 == 'T':
            print(word, pc2)
            
syllable = ['N', 'IH0', 'K', 'S']
i = [word for word, pron in entries if pron[-4:] == syllable]
print(i)
sorted(set(w[:2] for w, pron in entries if pron[0] == 'N' and w[0] != 'n'))

def stress(pron):
    return [char for phone in pron for char in phone if char.isdigit()]

#getting stress patterns   
i = [word for word, pron in entries if stress(pron) == ['0', '1', '0', '2', '0']]
print(set(i))
#%%
from nltk.corpus import swadesh
de2en = swadesh.entries(['de', 'en']) # German-English
es2en = swadesh.entries(['es', 'en']) # Spanish-English
translate = dict(de2en)
translate1 = dict(es2en)

translate['Mutter']

translate1['perro']
#%%
#wordnets are about networks of semantically related words
#nltk provides a module for this
from nltk.corpus import wordnet as wn
#the codes below retrieve synonyms
wn.synsets('boy')  #gets the synonym set for the specified string
"lemmas and lemma methods could be used"
#lemma_names in python 2 now == a method lemma_names() which returns a list of related words
#there is also a definition method which returns a description of the sense
#an example method which returns the usage of the word
wn.synset('male_child.n.01').lemma_names() #the pattern for synsets is
wn.synset('boy.n.02').lemma_names() #first the lemma or headword
wn.synset('son.n.01').definition() #the pos tag
wn.synset('boy.n.04').examples #then the sense number
wn.lemmas('dish') #returns a lemma list for dish

wn.synsets('automobile')
#%%
wn.synsets('kill')

wn.synset('car.n.01').hyponyms() #returns hyponyms of car

wn.synset('car.n.01').hypernyms() #returns the synset's hypernym

nltk.app.wordnet() #opens the inbuilt nltk wordnet dictionary

# .part/.subtance_meronyms() methods can be applied for meronymy
#.member_holonyms() for accesing holonymy
#.entailments() for entailment
#.antonyms() for antonymy works only on lemmas of a synset == wordnet.lemmas(synset).antonyms
#%%
#chap 3
from nltk.corpus import treebank
tree = treebank.parsed_sents('wsj_0002.mrg')[0]
tree.draw()
#%%
#analysing texts
from __future__ import division
import nltk, re, pprint
import urllib.request as ul

k = ul.urlopen("http://www.gutenberg.org/files/2554/2554.txt").read()
#len(k)
l = nltk.word_tokenize(k.decode())
#creating an nltk text file from token
token = nltk.Text(l)
#operations can be carried out on the text
print(token[:100])
token.collocations()
sen = nltk.pos_tag(token)
mamed = nltk.ne_chunk(sen[:10], binary=True)
mamed.draw()
#to accesss span specific parts of a text the the find and rfind methods are applied to the text object
k.find(b"PART I")
k.rfind(b"End of Project Gutenberg's Crime")
#after this the text can be appropriately sliced to carry out operations on specific regions
#%%
#working with HTML
from __future__ import division
import nltk, re, pprint
import urllib.request as ul
import bs4

#opening a raw html this returns a fully marked up text which is not ideal for analysis some pruning must be done
html = ul.urlopen("http://news.bbc.co.uk/2/hi/health/2284783.stm").read()
html1 = bs4.BeautifulSoup(html, "lxml")
clean = html1.text
#
#for tex in html1.body.find_all("div", class_="bodytext"):
#    text = tex.text
##the nltk.clean_html method comes in handy in preparing html for analysis
#it extracts the raw text from the html

w_tok = nltk.word_tokenize(clean)
s_tok = nltk.sent_tokenize(clean)
len(w_tok)
print(s_tok[:10])
tok = nltk.Text(w_tok)
tok.concordance("gene")
tok.count("gene")
tok.dispersion_plot(["gene", "blonde", "scientist"])
#%%
#working with rss feeds
from __future__ import division
import feedparser as fd
import nltk, re, pprint
import urllib.request as ul
import bs4

rss = fd.parse("http://languagelog.ldc.upenn.edu/nll/?feed=atom")
slice_ =  rss['feed']['title']
#print(slice_)

prin = rss.entries[2]
print(prin['title'])#prin.title is equivalent
#print(rss.keys())

#content = prin.content[0].value
#print(content[:70])
content = prin['content'][0]
print(content.values())
#%%
#working with files
#plaintext require no special methods to access them except for txt files in the nltk corpus
#which need be accesed though the
from __future__ import division
import feedparser as fd
import nltk, re, pprint

nltk.data.find('directory') #method
#binary files like doc files and pdf files have some third party dependencies
#like pywin and pypdf
#multicolumn documents pose a serious challenge
#%%
#dealing with string encodings
from __future__ import division
import feedparser as fd
import nltk, re, pprint, codecs
#the codecs lobrary provides a way of dealing with encodings
f = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
k = codecs.open(f, encoding='latin2')

for line in k:
    line = line.strip()
    print(lines.encode('unicode_escape'))
#%%
import re
j = "I loved you yet you decided to say she is redder"

re.findall(r'ed', j)

re.findall(r'ed\b', j)

re.sub(r'ed\b', r'es',j)
#%%
from __future__ import division
import feedparser as fd
import nltk, re, pprint, codecs
from nltk.corpus import words

i = [w for w in words.words('en') if re.findall(r'ed\b', w)]

wordlist = [w for w in words.words('en') if w.islower()]

for word in wordlist:
    if re.search(r'^..j..t..$', word):
        print(word)
#%%
#regex ranges denoted by []
i = [print(w) for w in wordlist if re.search(r'^[ghi][mno][jlk][def]$', w)]
j = print(list([i for i in wordlist if re.search(r'^[^aeiouAEIOU]+$', i)]))
#%%
#print([int(i) for i in re.findall(r'\d+', '2009-12-31')])
#%%
import nltk
import re
from nltk.corpus import treebank

wsj = sorted(set(treebank.words()))

fd = nltk.FreqDist([vs for word in wsj for vs in re.findall(r'[aeiou]{2,}', word)])
print(fd.items())
fd.plot(10, cumulative=True)
#%%
regexp = r'^[AEIOUaeiou]+|[AEIOUaeiou]+$|[^AEIOUaeiou]'
def con(word):
    part = re.findall(regexp, word)
    return ''.join(part)

print(nltk.tokenwrap([con(w) for w in ['I', 'Am', 'hungry']])) #nltk.tokenwrap returns a nicely fomatted streem of strings
#%%
from nltk.corpus import toolbox

#k = toolbox.words('rotokas.dic')
#cfd = [cv for w in k for cv in re.findall(r'[pkrsvt][aeiou]', w)]
#cfd1 = nltk.ConditionalFreqDist(cfd)
#cfd1.tabulate()
##the wurds within which the mathing expressions could be accessed through nltk.index command
#cfd2 = [(w, cv) for cv in k for w in re.findall(r'[pkrsvt][aeiou]', cv)]
#ind = nltk.Index(cfd2)
#ind['ti']

txt = """Agbi o, anẹtẹ, ogha mẹ na ririri asi. Ọmami ọkpa u nẹ ghi, ghi ọwa tse edio ọ va muzomi, ni ọni ọmami na ọ tsẹni nọva muzomi ọdọ nọ ne muzomi ọ ma nẹ enini kọle. U nẹ ghi ọ ma me ghi ọwa muzomi 
na, ọgwozo ye emale, ọdọ ni ma le, ọ ma me o. Ọ ma ghini agbọ ni a ki ye na, khi ọmọze ọ ẹ ọle ma lemale, ọgwozo rẹ le. Agbọ ni ediọ ghi ọgwozo ye emale ni ọmọze ma le ha ẹmọni nọ o!, uvoso na a 
ghi: ni oni ọgwozo, ni ọ a dẹ ọkhọkhọ, la ọkhọkhọ li ughọ, nọ va rɛ riama ọdọni. Ọni ọmami na ọ riama ọdọni riama riama, ọ ma le, ọ ghi ọle kue le emale kͻle adabi ghi ọ ghi ọ nẹ enini kͻle, Ni ọ
gu enini ma ni ọgwozo na ọ ma yama gu ma ni. Ọni ọgwozo na ki sẹnẹ ẹdọ ọ vọ ikpo, afiamin ọkpa ki tize a so iyolo okpa: ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami 
nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọ ma nẹ enini ọdọ, ri ema ze, ri ema le, Iyezumegha a tie ọni o suẹ suẹ suẹ.  Ilọ ọli lu, ilọ ọli gha ye emale ọ ma le ọ viẹ viẹ viẹ 
viẹ viẹ. Ẹlẹ ọkpa na ni ọ sẹnẹ ẹdoọ, ọni afiamin ki da gbolo aze a so iyolo: ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami nọ vọ ikpo ẹdọ suẹ suẹ suẹ, ọmami nọ vọ ikpo
ẹdọ suẹ suẹ suẹ, ọ ma nẹ enini ọdọ, ri ema ze, ri ema le, Iyezumegha a tie ọni o suẹ suẹ suẹ. Eeh! ọ da wẹmẹ ọ da sẹnẹ owa ọ ye ozọmi ki ikpigba na bẹnọ ọni, ọla ọ nẹ ikpigba na bẹnọ ọni?, ọla ọ 
nɛ emi e a ti ikpigba aze na? ikpigba na bẹnọ ọni, ọ họn ọni, ọni ikpigba na bẹnọ ọni na a ma gha ri amẹ zhọ o!, a ma ri amẹ zhọ, a ri alubasa zhọ, a ri ighuzumu zho, ọna ti ighuzumu na u dọ rɛ 
ye ozọmi ọ va tue kua zhi ẹkẹli ozọmi. Ọ rumu ema zhọ bẹnẹ bẹnẹ, du ọni zhi otọ ni ọdọ ni gayi!. ọdọ ni ọle ọlele ̣le ma le. Ọni ọgwozo ẹ Iyesumegha ghi enini kẹ, ọni ọmọze da wẹmẹ, e da yeni agbọ
kale weli ewẹmẹ. Ilọ nọ o. Agbi o anɛtɛ ogha mẹ na ririri a sui."""

cf = nltk.ConditionalFreqDist([vc for w in txt.split(" ") for vc in re.findall(r'[bcdfghjklmnpqrstvwyz][AEIOUaeiou]', w)])
cf.tabulate(5)

k = txt.encode('utf-8')
print(k.decode('utf-8'))
#%%
#() denote groups in regex and the class of items to be extracted
#finding word stems with regex
#extracting only the affix
print(re.findall(r'^.*(es|s|ed|ing|ly|ious|ies|ive|ment)$', 'pretty little faces')) #returns s
#to extract both stem and affix jointly
print(re.findall(r'^.*(?:es|s|ed|ing|ly|ious|ies|ive|ment)$', 'facing')) #returns facing
#to extract stem and affix in separate tuple groups
print(re.findall(r'^(.*?)(es|s|ed|ing|ly|ious|ies|ive|ment)?$', 'face'))
#to extract only the stem with regex
print(re.findall(r'^(.*?)es|s|ed|ing|ly|ious|ies|ive|ment$', 'faces'))

def stemmer(args):
    k = nltk.word_tokenize(args)
    j = [(print(stem), aff) for i in k for stem, aff in re.findall(r'^(.*?)(es|s|ed|ing|ly|ious|ies|ive|ment)?$', i)]
#%%
#nltk comes with an inbuilt findall method relevent for accessing parts of texts
from nltk.corpus import gutenberg, nps_chat
wsj = gutenberg.words('melville-moby_dick.txt')
tek =  nltk.Text(wsj)
tek.findall(r"<a>(<.*>)<man>")

heybro = nltk.Text(nps_chat.words())
heybro.findall(r'<.*><.*><bro>')
heybro.findall(r'<l.*>{3,}')

nltk.re_show(r'\b\w{3,5}\b', "I am Jesse I need you to help me get accross to the other side") #this marksup all matching outputs
re.findall(r'[A-Z][a-z]{2,}', k)
#%%
#tokenizing text withregex
#the re.split() method comes in handy for breaking strings into pieces
k = "I am Jesse I need you to help me get accross to the other side"
re.split('\W+', k) #returns only valid strings prunes digits
#accesses words with hyphens within
nltk.re_show(r'\w+(?:-+\w*)', token)
#%%
#segmenting unsegmented text this prooves particulartly challenging in NLP
#especially in ideographic languages howerver some clever reasoning could get us through the hurdle

sent = "IlovetoeatcarrotsalotIlearntthattheyaregoodfortheskinPleasemyfriendseatcarrots"
seg1 = "100010100100000011001100000100010001001000100100100010000010100000010010000001"
seg2 = "000000000000000000001000000000000000000000000000000010000000000000000000000001"
def segment(text, seg):
    parsed = []
    last = 0
    for i in range(len(seg)):
        if seg[i] == '1':
            parsed.append(text[last:i+1])
            last = i + 1
    parsed.append(text[last:])
    return parsed
    
#%%
import nltk
from nltk.stem import WordNetLemmatizer
jay = "I am Jesse I need you to help, I ate chicken yesterday, and i am eating chicken now me get accross to the other side"
lemma = WordNetLemmatizer()
print("{:^30}".format("Tokenised Words"))
print("{:<10}{:>20}".format("Stemmed Words", "Lemmatized Words"))
for i in nltk.word_tokenize(jay):
    print("{:<10}{:>20}".format(nltk.PorterStemmer().stem(i), lemma.lemmatize(i, pos="v")))
#lemma.lemmatize("eating")
"""
for item in reversed(s) Iterate over elements of s in reverse
for item in set(s).difference(t) Iterate over elements of s not in t
for item in random.shuffle(s) Iterate over elements of s in random order
"""
#%%
from nltk.corpus import brown
word = brown.words()
freq = nltk.FreqDist(word)
cum = 0.0

for index, word in enumerate(freq):
    cum += freq[word] * 100 / freq.N()
    print("%3d %6.2f%% %s"% (index+1, cum, word))
    if cum == 25:
        break
#%%
#the pdb module is important for debugging code
import pdb
pdb.run("thefunction(param)")
#after which methods of the pdb like args, next, step can be called