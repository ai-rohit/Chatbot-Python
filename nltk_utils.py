import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# PorterStemmer for stemming words
stemmer = PorterStemmer()


# tokenize sentence using nltk
def tokenize(input_sentence):
    return nltk.word_tokenize(input_sentence)


# stem words
def stem(words):
    return stemmer.stem(words.lower())


# create bag of tokens for input
def bag_of_words(tokenize_sentence,all_words):
    tokenize_sentence = [stem(word) for word in tokenize_sentence]
    bag =  np.zeros(len(all_words), dtype=np.float32)
    for index, word in enumerate(all_words):
        if word in tokenize_sentence:
            bag[index] = 1.0
    return bag
