#Practical No. 10: Speech Tagging, Statistical, Probabilistic & Malt Parsing

#10(a)(i) Speech Tagging Using spaCy

import spacy
from spacy import displacy

sp = spacy.load('en_core_web_sm')
sen = sp("I like to play football. I hated it in my childhood though")
print(sen.text)
print(sen[7].pos_)
print(sen[7].tag_)
print(spacy.explain(sen[7].tag_))

for word in sen:
    print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')

sen_g1 = sp('Can you google it?')
word = sen_g1[2]
print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')

sen_g2 = sp('Can you search it on google?')
word = sen_g2[5]
print(f'{word.text:{12}} {word.pos_:{10}} {word.tag_:{8}} {spacy.explain(word.tag_)}')

num_pos = sen.count_by(spacy.attrs.POS)
for k, v in sorted(num_pos.items()):
    print(f'{k}. {sen.vocab[k].text:{8}}: {v}')

displacy.serve(sen, style='dep', options={'distance': 120})






#10(a)(ii) Speech Tagging Using NLTK

import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

nltk.download('state_union')
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:2]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))

process_content()







#10(b)(i) Statistical Parsing

import nltk

nltk.download('treebank')

def give(t):
    return (
        t.label() == 'VP'
        and len(t) > 2
        and t[1].label() == 'NP'
        and (t[2].label() == 'PP-DTV' or t[2].label() == 'NP')
        and ('give' in t[0].leaves() or 'gave' in t[0].leaves())
    )

def sent(t):
    return ' '.join(token for token in t.leaves() if token[0] not in '*-0')

def print_node(t, width):
    output = "%s %s: %s/%s: %s" % (
        sent(t[0]), t[1].label(),
        sent(t[1]), t[2].label(), sent(t[2])
    )
    if len(output) > width:
        output = output[:width] + "..."
    print(output)

for tree in nltk.corpus.treebank.parsed_sents():
    for t in tree.subtrees(give):
        print_node(t, 72)







#10(b)(ii) Probabilistic Parser Using PCFG & ViterbiParser

import nltk
from nltk import PCFG

grammar = PCFG.fromstring("""
NP -> NNS [0.5] | JJ NNS [0.3] | NP CC NP [0.2]
NNS -> "men" [0.1] | "women" [0.2] | "children" [0.3] | NNS CC NNS [0.4]
JJ -> "old" [0.4] | "young" [0.6]
CC -> "and" [0.9] | "or" [0.1]
""")
print(grammar)

viterbi_parser = nltk.ViterbiParser(grammar)
token = "old men and women".split()
obj = viterbi_parser.parse(token)
print("Output: ")
for x in obj:
    print(x)






#10(c) Malt Parsing

from nltk.parse import malt

mp = malt.MaltParser('maltparser-1.7.2', 'engmalt.linear-1.7.mco')
t = mp.parse_one('I saw a bird from my window.'.split()).tree()
print(t)
t.draw()















        
