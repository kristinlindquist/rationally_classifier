import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import numpy as np
from stop_words import get_stop_words

stop = get_stop_words('en')
stop.append("Change")
stop.append("Baseline")

lemmatizer = WordNetLemmatizer() 
detokenizer = TreebankWordDetokenizer()

def handle_punct(w_list):
    return [word for word in w_list if word.isalpha()]

def get_desired_pos(w_list):
    # get desired parts-of-speech: noun, number, adjective, verb or unknown
    poses = ('NOUN', 'X', 'NUM');
    return [word for (word, pos) in nltk.pos_tag(w_list, tagset='universal') if pos in poses]

def lem(w_list):
    return [lemmatizer.lemmatize(word) for word in w_list]

def handle_stop_words(w_list):
    return [word for word in w_list if word not in lem(stop)]

def preprocess_sentence(s):
    s = s.lower() if s else ""
    return s.strip()

def preprocess_word(s):
  if not s:
    return None

  w_list = word_tokenize(s)
  w_list = handle_punct(w_list)
  w_list = get_desired_pos(w_list)
  w_list = lem(w_list)
  w_list = handle_stop_words(w_list)
  return detokenizer.detokenize(w_list);

def preprocess(df):
    print('Preprocessing sentences ...')
    sentences = df.apply(lambda row: preprocess_sentence(row[0]), axis=1)
    topics = df.apply(lambda row: row[1], axis=1)

    print('Done preprocessing.')
    return sentences, topics
