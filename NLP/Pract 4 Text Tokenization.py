#Practical No. 4: Text Tokenization

text = """This tool is an a beta stage. Alexa developers can use Get Metrics API to
seamlessly analyse metric. It also supports custom skill model, prebuilt Flash Briefing
model, and the Smart Home Skill API. You can use this tool for creation of monitors,
alarms, and dashboards that spotlight changes. The release of these three tools will
enable developers to create visual rich skills for Alexa devices with screens. Amazon
describes these tools as the collection of tech and tools for creating visually rich and
interactive voice experiences."""

data = text.split('.')
for i in data:
    print(i)






#4(b) Regular Expressions

from nltk.tokenize import RegexpTokenizer

tk = RegexpTokenizer(r'\s+', gaps=True)
str_input = "I love to study Natural Language Processing in Python"
tokens = tk.tokenize(str_input)
print(tokens)






#4(c) NLTK Tokenization

from nltk.tokenize import word_tokenize

str_input = "I love to study Natural Language Processing in Python"
print(word_tokenize(str_input))






#4(d) spaCy Tokenization

import spacy

nlp = spacy.blank("en")
str_input = "I love to study Natural Language Processing in Python"
doc = nlp(str_input)
words = [word.text for word in doc]
print(words)






#4(e) Keras Tokenization

from tensorflow.keras.preprocessing.text import text_to_word_sequence

str_input = "I love to study Natural Language Processing in Python"
tokens = text_to_word_sequence(str_input)
print(tokens)






#4(f) Gensim Tokenization

from gensim.utils import tokenize

str_input = "I love to study Natural Language Processing in Python"
print(list(tokenize(str_input)))



