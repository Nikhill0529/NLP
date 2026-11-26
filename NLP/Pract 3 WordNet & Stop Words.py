#Practical No. 3: WordNet & Stop Words


#3(a) WordNet Dictionary

import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

print(wordnet.synsets("computer"))
print(wordnet.synset("computer.n.01").definition())
print("Examples:", wordnet.synset("computer.n.01").examples())
print(wordnet.lemma('buy.v.01.buy').antonyms())







#3(b) Lemmas, Hyponyms & Hypernyms

import nltk
from nltk.corpus import wordnet

print(wordnet.synsets("computer"))
print(wordnet.synset("computer.n.01").lemma_names())

for e in wordnet.synsets("computer"):
    print(f'{e} --> {e.lemma_names()}')

print(wordnet.synset('computer.n.01').lemmas())
print(wordnet.lemma('computer.n.01.computing_device').synset())
print(wordnet.lemma('computer.n.01.computing_device').name())

syn = wordnet.synset('computer.n.01')
print(syn.hyponyms())
print([lemma.name() for synset in syn.hyponyms() for lemma in synset.lemmas()])

vehicle = wordnet.synset('vehicle.n.01')
car = wordnet.synset('car.n.01')
print(car.lowest_common_hypernyms(vehicle))








#3(c) Synonym & Antonym of "active"


from nltk.corpus import wordnet

print(wordnet.synsets("active"))
print(wordnet.lemma('active.a.01.active').antonyms())







#3(d) Path Similarity

import nltk
from nltk.corpus import wordnet

syn1 = wordnet.synsets('football')
syn2 = wordnet.synsets('soccer')

for s1 in syn1:
    for s2 in syn2:
        print("Path similarity of: ")
        print(s1, '(', s1.pos(), ')', '[', s1.definition(), ']')
        print(s2, '(', s2.pos(), ')', '[', s2.definition(), ']')
        print(" is", s1.path_similarity(s2))
        print()








#3(e)(i) Stop Words in NLTK

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
text = "Yashesh likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if word not in stopwords.words('english')]
print(tokens_without_sw)

all_stopwords = stopwords.words('english')
all_stopwords.append('play')
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
print(tokens_without_sw)

all_stopwords.remove('not')
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
print(tokens_without_sw)






#3(e)(ii) Stop Words in Gensim

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from nltk.tokenize import word_tokenize

text = "Yashesh likes to play football, however he is not too fond of tennis."
filtered_sentence = remove_stopwords(text)
print(filtered_sentence)

all_stopwords = gensim.parsing.preprocessing.STOPWORDS
print(all_stopwords)

all_stopwords_gensim = STOPWORDS.union(set(['likes', 'play']))
text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords_gensim]
print(tokens_without_sw)

sw_list = {"not"}
all_stopwords_gensim = STOPWORDS.difference(sw_list)
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords_gensim]
print(tokens_without_sw)






#3(e)(iii) Stop Words in spaCy

import spacy
from nltk.tokenize import word_tokenize

sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
all_stopwords.add("play")

text = "Yashesh likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(text)
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
print(tokens_without_sw)

all_stopwords.remove('not')
tokens_without_sw = [word for word in text_tokens if word not in all_stopwords]
print(tokens_without_sw)




