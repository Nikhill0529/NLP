#Practical No. 2: Corpora, CFD, POS Tagging & Word Segmentation
# 2(a) Brown Corpus
import nltk
from nltk.corpus import brown

print('File ids of brown corpus\n', brown.fileids())
ca01 = brown.words('ca01')
print('\nca01 has following words: \n', ca01)
print('\nca01 has', len(ca01), 'words')
print('\n\nCategories or file in brown corpus: \n')
print(brown.categories())
print('\n\nStatistics for each text:\n')
print('AvgWordLen\tAvgSentenceLen\tno.ofTimesEachWordAppearsOnAvg\t\tFileName')

for fileid in brown.fileids():
    num_chars = len(brown.raw(fileid))
    num_words = len(brown.words(fileid))
    num_sents = len(brown.sents(fileid))
    num_vocab = len(set([w.lower() for w in brown.words(fileid)]))
    print(
        int(num_chars / num_words),
        '\t\t\t',
        int(num_words / num_sents),
        '\t\t\t',
        int(num_words / num_vocab),
        '\t\t\t',
        fileid,
    )







#2(b) Custom Plaintext Corpus


import nltk
from nltk.corpus import PlaintextCorpusReader

corpus_root = 'D:/2020/NLP/Practical/uni'
filelist = PlaintextCorpusReader(corpus_root, '.*')
print('\n File list: \n')
print(filelist.fileids())
print(filelist.root)
print('\n\nStatistics for each text:\n')
print('AvgWordLen\tAvgSentenceLen\tno.ofTimesEachWordAppearsOnAvg\tFileName')

for fileid in filelist.fileids():
    num_chars = len(filelist.raw(fileid))
    num_words = len(filelist.words(fileid))
    num_sents = len(filelist.sents(fileid))
    num_vocab = len(set([w.lower() for w in filelist.words(fileid)]))
    print(
        int(num_chars / num_words),
        '\t\t\t',
        int(num_words / num_sents),
        '\t\t\t',
        int(num_words / num_vocab),
        '\t\t',
        fileid,
    )







#2(c) Conditional Frequency Distributions

import nltk
from nltk.corpus import brown, inaugural, udhr

fd = nltk.ConditionalFreqDist(
    (genre, word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

genre_word = [
    (genre, word)
    for genre in ['news', 'romance']
    for word in brown.words(categories=genre)
]
print(len(genre_word))
print(genre_word[:4])
print(genre_word[-4:])

cfd = nltk.ConditionalFreqDist(genre_word)
print(cfd)
print(cfd.conditions())
print(cfd['news'])
print(cfd['romance'])
print(list(cfd['romance']))

cfd_inaug = nltk.ConditionalFreqDist(
    (target, fileid[:4])
    for fileid in inaugural.fileids()
    for w in inaugural.words(fileid)
    for target in ['america', 'citizen']
    if w.lower().startswith(target)
)

languages = [
    'Chickasaw', 'English', 'German_Deutsch',
    'Greenlandic_Inuktikut', 'Hungarian_Magyar', 'Ibibio_Efik'
]
cfd_udhr = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang + '-Latin1')
)
cfd_udhr.tabulate(conditions=['English', 'German_Deutsch'], samples=range(10), cumulative=True)









#2(d) Sentence and Word Tokenization

import nltk
from nltk import tokenize

nltk.download('punkt')
nltk.download('words')

para = "Hello! My name is mcc. Today you'll be learning NLTK."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================\n", sents)
print("\nword tokenization\n===================\n")
for index in range(len(sents)):
    words = tokenize.word_tokenize(sents[index])
    print(words)








#2(e) Most Frequent Noun Tags

import nltk
from collections import defaultdict

nltk.download('averaged_perceptron_tagger')
text = nltk.word_tokenize("Nick likes to play football. Nick does not like to play cricket.")
tagged = nltk.pos_tag(text)
print(tagged)

addNounWords = []
for word, tag in tagged:
    if tag in ('NN', 'NNS', 'NNPS', 'NNP'):
        addNounWords.append(word)

print(addNounWords)

temp = defaultdict(int)
for sub in addNounWords:
    for wrd in sub.split():
        temp[wrd] += 1

res = max(temp, key=temp.get)
print("Word with maximum frequency: " + str(res))








#2(f) Python Dictionaries

thisdict = {
    "brand": "Ford",
    "model": "Mustang",
    "year": 1964
}
print(thisdict)
print(thisdict["brand"])
print(len(thisdict))
print(type(thisdict))







#2(g)(i) DefaultTagger

import nltk
from nltk.tag import DefaultTagger
from nltk.corpus import treebank

exptagger = DefaultTagger('NN')
testsentences = treebank.tagged_sents()[1000:]
print(exptagger.evaluate(testsentences))
print(exptagger.tag_sents([['Hi', ','], ['How', 'are', 'you', '?']]))








#2(g)(ii) Regular Expression Tagger

from nltk.corpus import brown
from nltk.tag import RegexpTagger

test_sent = brown.sents(categories='news')[0]
regexp_tagger = RegexpTagger([
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),
    (r'^(The|the|A|a|An|an)$', 'AT'),
    (r'.*able$', 'JJ'),
    (r'.*ness$', 'NN'),
    (r'.*ly$', 'RB'),
    (r'.*s$', 'NNS'),
    (r'.*ing$', 'VBG'),
    (r'.*ed$', 'VBD'),
    (r'.*', 'NN')
])
print(regexp_tagger)
print(regexp_tagger.tag(test_sent))








#2(g)(iii) Unigram Tagger

from nltk.tag import UnigramTagger
from nltk.corpus import treebank

train_sents = treebank.tagged_sents()[:10]
tagger = UnigramTagger(train_sents)
print(treebank.sents()[0])
print('\n', tagger.tag(treebank.sents()[0]))

tagger = UnigramTagger(model={'Pierre': 'NN'})
print('\n', tagger.tag(treebank.sents()[0]))








#2(h) Word Segmentation Without Spaces

import re

words = []
testword = []

print("MENU")
print("---------------------------------------")
print(" 1. Hash tag segmentation")
print(" 2. URL segmentation ")
print("---------------------------------------")
print("enter the input choice for performing word segmentation")
choice = int(input())

if choice == 1:
    text = "#whatismyname"
    print("input with HashTag", text)
    pattern = re.compile(r"[^\w']")
    a = pattern.sub('', text)
elif choice == 2:
    text = "www.whatismyname.com"
    print("input with URL", text)
    a = re.split(r'\s|(?<!\d)[,.](?!\d)', text)
    splitwords = ["www", "com", "in"]
    a = "".join([each for each in a if each not in splitwords])
else:
    print("wrong choice...try again")
    a = ""

print(a)
for each in a:
    testword.append(each)
test_lenth = len(testword)

with open('words.txt', 'r') as f:
    lines = f.readlines()
words = [(e.strip()) for e in lines]

def Seg(a, lenth):
    ans = []
    for k in range(0, lenth + 1):
        if a[0:k] in words:
            print(a[0:k], "-appears in the corpus")
            ans.append(a[0:k])
            break
    if ans:
        g = max(ans, key=len)
        return g
    return 0

test_tot_itr = 0
answer = []
N = 37
while test_tot_itr < test_lenth:
    ans_words = Seg(a, test_lenth)
    if not ans_words:
        break
    print(ans_words)
    a = a[len(ans_words):]
    test_tot_itr += len(ans_words)









    
