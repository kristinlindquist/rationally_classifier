import nltk
from nltk.stem.porter import *
from nltk import bigrams
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

stop = get_stop_words('en')
stop.append('administer')
stop.append('assess')
stop.append('blinded')
stop.append('change')
stop.append('clinical')
stop.append('cohort')
stop.append('combination')
stop.append('comparison')
stop.append('control')
stop.append('controlled')
stop.append('disease')
stop.append('disorder')
stop.append('dose')
stop.append('dosage')
stop.append('drug')
stop.append('effect')
stop.append('efficacy')
stop.append('effectiveness')
stop.append('evaluate')
stop.append('evaluation')
stop.append('feasibility')
stop.append('impact')
stop.append('intervention')
stop.append('manage')
stop.append('management')
stop.append('measure')
stop.append('measurement')
stop.append('multicenter')
stop.append('outcome')
stop.append('participant')
stop.append('patient')
stop.append('pharmacodynamics')
stop.append('pharmacokinetics')
stop.append('phase')
stop.append('pilot')
stop.append('placebo')
stop.append('procedure')
stop.append('program')
stop.append('prospective')
stop.append('randomized')
stop.append('response')
stop.append('safety')
stop.append('study')
stop.append('subject')
stop.append('surveillance')
stop.append('symptom')
stop.append('syndrome')
stop.append('therapy')
stop.append('tolerability')
stop.append('treatment')
stop.append('trial')
stop.append('use')
stop.append('versus')
stop.append('volunteer')

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
    w_list = handle_punct(w_list)
    w_list = get_desired_pos(w_list)
    w_list = stem(w_list)
    w_list = handle_stop_words(w_list)
    return w_list;
