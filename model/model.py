from Autoencoder import *
from datetime import datetime
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from preprocess import *
import numpy as np
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans
from transformers import *

def preprocess(docs, samp_size=100):
    print('Preprocessing data ...')
    n_docs = len(docs)
    sentences = []
    token_lists = []
    samp = np.random.choice(n_docs, samp_size)
    for i, idx in enumerate(samp):
        sentence = preprocess_sentence(docs[idx])
        sentences.append(sentence)
        token_lists.append(preprocess_word(sentence))
        print('{} %'.format(str(np.round((i + 1) / len(samp) * 100, 2))), end='\r')

    print('Done preprocessing.')
    return sentences, token_lists


def get_vec_lda(model, corpus, k, topics=None):
    n_doc = len(corpus)
    vec_lda = np.zeros((n_doc, k))
    for i in range(n_doc):
        topic_probs = model.get_document_topics(corpus[i]) if topics is None else model[corpus[i]]
        for topic, prob in topic_probs:
            if topics is None or topic in topics:
                vec_lda[i, topic] = prob

    return vec_lda


class Topic_Model:
    def __init__(self, k=10):
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.lda_model = None
        self.method = 'LDA_BERT'
        self.vec = {}
        self.gamma = 15
        self.autoencoder = None
        self.id = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def vectorize(self, sentences, token_lists, method=None):
        if method is None:
            method = self.method

        self.dictionary = corpora.Dictionary(token_lists)
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]

        if method == 'LDA':
            print('LDA vectorizing...')
            self.ldamodel = LdaModel(
                self.corpus,
                num_topics=self.k,
                id2word=self.dictionary,
                passes=20
            )

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Done LDA vectorizing.')
            return vec

        elif method == 'BERT':
            print('Vectorizing for BERT ...')

            # word_embedding_model = models.Transformer('bert-base-uncased')
            word_embedding_model = models.Transformer('emilyalsentzer/Bio_ClinicalBERT')

            # Apply mean pooling to get one fixed sized sentence vector
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False
            )

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Done BERT vectorizing.')
            return vec

    def meta_vectorize(self, sentences, token_lists):
        vec_lda = self.vectorize(
            sentences,
            token_lists,
            method='LDA'
        )
        vec_bert = self.vectorize(
            sentences,
            token_lists,
            method='BERT'
        )
        vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
        self.vec['LDA_BERT_FULL'] = vec_ldabert
        self.autoencoder = Autoencoder()
        print('Fitting Autoencoder ...')
        self.autoencoder.fit(vec_ldabert)
        print('Done fitting autoencoder')
        
        return self.autoencoder.encoder.predict(vec_ldabert)

    def fit(self, sentences, token_lists):
        m_clustering = KMeans

        print('Clustering ...')
        self.cluster_model = m_clustering(self.k)
        self.vec[self.method] = self.meta_vectorize(sentences, token_lists)
        self.cluster_model.fit(self.vec[self.method])
        print('Done clustering.')

    def predict(self, sentences, token_lists, out_of_sample=None):
        if out_of_sample is not None:
            corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            vec = self.vectorize(sentences, token_lists)
        else:
            corpus = self.corpus
            vec = self.vec.get(self.method, None)

        return self.cluster_model.predict(vec)
