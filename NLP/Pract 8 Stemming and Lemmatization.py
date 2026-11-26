#Practical No. 8: Stemming and Lemmatization

#8(a) PorterStemmer

from nltk.stem import PorterStemmer

word_stemmer = PorterStemmer()
print(word_stemmer.stem('writing'))




#8(b) LancasterStemmer

from nltk.stem import LancasterStemmer

Lanc_stemmer = LancasterStemmer()
print(Lanc_stemmer.stem('writing'))




#8(c) RegexpStemmer

from nltk.stem import RegexpStemmer

Reg_stemmer = RegexpStemmer('ing$|s$|able$', min=4)
print(Reg_stemmer.stem('writing'))




#8(d) SnowballStemmer

from nltk.stem import SnowballStemmer

english_stemmer = SnowballStemmer('english')
print(english_stemmer.stem('writing'))




#8(e) WordNetLemmatizer

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print("word:\tlemma")
print("rocks:", lemmatizer.lemmatize("rocks"))
print("corpora:", lemmatizer.lemmatize("corpora"))
print("better:", lemmatizer.lemmatize("better", pos="a"))







