import nltk
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
import numpy as np
from stop_words import get_stop_words

stop = get_stop_words('en')

stemmer = PorterStemmer()

def handle_punct(w_list):
    return [word for word in w_list if word.isalpha()]

def get_desired_pos(w_list):
    # get desired parts-of-speech: noun, number, adjective, verb or unknown
    poses = ('NOUN', 'ADJ', 'VERB', 'ADV', 'X', 'NUM');
    return [word for (word, pos) in nltk.pos_tag(w_list, tagset='universal') if pos in poses]

def stem(w_list):
    return [stemmer.stem(word) for word in w_list]

def handle_stop_words(w_list):
    return [word for word in w_list if word not in stem(stop)]

def preprocess_sentence(s):
    s = s.lower() if s else ""
    return s.strip()

def preprocess_word(s):
    if not s:
        return None

    w_list = word_tokenize(s)
    # w_list = handle_punct(w_list)
    # w_list = get_desired_pos(w_list)
    # w_list = stem(w_list)
    # w_list = handle_stop_words(w_list)
    return w_list;

def preprocess(df):
    print('Preprocessing sentences ...')
    sentences = df.apply(lambda row: preprocess_sentence(row[0]), axis=1)
    topics = df.apply(lambda row: row[1], axis=1)

    print('Preprocessing measurements ...')
    m_names = df.apply(lambda row: preprocess_sentence(row[2]), axis=1)

    print('Done preprocessing.')
    return sentences, topics, m_names
