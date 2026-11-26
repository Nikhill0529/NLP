#Practical No. 6: POS Tagging, Chunking & NER

#6(a) Tokenization, POS Tagging & Chunking

import nltk
from nltk import tokenize, tag, chunk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

para = "Hello! My name is mcc. Today you'll be learning NLTK."
sents = tokenize.sent_tokenize(para)
print("\nsentence tokenization\n===================\n", sents)
print("\nword tokenization\n===================\n")

words_list = []
for index in range(len(sents)):
    words = tokenize.word_tokenize(sents[index])
    words_list.append(words)
    print(words)

tagged_words = []
for index in range(len(sents)):
    tagged_words.append(tag.pos_tag(words_list[index]))
print("\nPOS Tagging\n===========\n", tagged_words)

tree = []
for index in range(len(sents)):
    tree.append(chunk.ne_chunk(tagged_words[index]))
print("\nchunking\n========\n")
print(tree)







#6(b) Named Entity & Syntax Analysis with spaCy

import spacy

nlp = spacy.load("en_core_web_sm")
text = (
    "When Sebastian Thrun started working on self-driving cars at "
    "Google in 2007, few people outside of the company took him "
    "seriously. \"I can tell you very senior CEOs of major American "
    "car companies would shake my hand and turn away because I wasn't "
    "worth talking to,\" said Thrun, in an interview with Recode earlier "
    "this week."
)
doc = nlp(text)
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])






#6(c) Treebank Chunking Diagram

import nltk
from nltk.corpus import treebank_chunk

nltk.download('treebank')
print(treebank_chunk.tagged_sents()[0])
print(treebank_chunk.chunked_sents()[0])
treebank_chunk.chunked_sents()[0].draw()






