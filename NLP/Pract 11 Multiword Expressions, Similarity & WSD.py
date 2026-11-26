#Practical No. 11: Multiword Expressions, Similarity & WSD

#11(a) Multiword Expressions

from nltk.tokenize import MWETokenizer
from nltk import sent_tokenize, word_tokenize

s = '''Good cake cost Rs.1500 kg in Mumbai. Please buy me one of them.\n\nThanks.'''
mwe = MWETokenizer([('New', 'York'), ('Hong', 'Kong')], separator='_')
for sent in sent_tokenize(s):
    print(mwe.tokenize(word_tokenize(sent)))








#11(b) Normalized Web Distance & Word Similarity


import re
import numpy as np
import textdistance
from sklearn.cluster import AgglomerativeClustering

texts = [
    'Reliance supermarket', 'Reliance hypermarket', 'Reliance', 'Reliance',
    'Reliance downtown', 'Relianc market',
    'Mumbai', 'Mumbai Hyper', 'Mumbai dxb', 'mumbai airport',
    'k.m trading', 'KM Trading', 'KM trade', 'K.M. Trading', 'KM. Trading'
]

def normalize(text):
    """Keep only lower-cased text and numbers"""
    return re.sub('[^a-z0-9]+', '', text.lower())

def group_texts(texts, threshold=0.4):
    """Replace each text with the representative of its cluster"""
    normalized_texts = np.array([normalize(text) for text in texts])
    distances = 1 - np.array([
        [textdistance.jaro_winkler(one, another) for one in normalized_texts]
        for another in normalized_texts
    ])
    clustering = AgglomerativeClustering(
        distance_threshold=threshold,
        metric="precomputed",
        linkage="complete",
        n_clusters=None
    ).fit(distances)

    centers = dict()
    for cluster_id in set(clustering.labels_):
        index = clustering.labels_ == cluster_id
        centrality = distances[:, index][index].sum(axis=1)
        centers[cluster_id] = normalized_texts[index][centrality.argmin()]

    return [centers[i] for i in clustering.labels_]

print(group_texts(texts))









#11(c) Word Sense Disambiguation

from nltk.corpus import wordnet as wn

def get_first_sense(word, pos=None):
    if pos:
        synsets = wn.synsets(word, pos)
    else:
        synsets = wn.synsets(word)
    return synsets[0]

best_synset = get_first_sense('bank')
print('%s: %s' % (best_synset.name(), best_synset.definition()))

best_synset = get_first_sense('set', 'n')
print('%s: %s' % (best_synset.name(), best_synset.definition()))

best_synset = get_first_sense('set', 'v')
print('%s: %s' % (best_synset.name(), best_synset.definition()))







