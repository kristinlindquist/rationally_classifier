from Autoencoder import *
from nnclassifier import *
from datetime import datetime
import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from preprocess import *
from sentence_transformers import models, SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import *

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
    def __init__(self, k):
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.lda_model = None
        self.model = None
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
            word_embedding_model = models.Transformer('emilyalsentzer/Bio_ClinicalBERT')

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

    def meta_vectorize(self, sentences, token_lists, vec_bert):
        vec_lda = self.vectorize(
            sentences,
            token_lists,
            method='LDA'
        )
        vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
        self.vec['LDA_BERT_FULL'] = vec_ldabert
        self.autoencoder = Autoencoder()
        print('Fitting Autoencoder ...')
        self.autoencoder.fit(vec_ldabert)
        print('Done fitting autoencoder')
        
        return self.autoencoder.encoder.predict(vec_ldabert)

    def fit(self, sentences, token_lists, topics):
        m_clustering = KMeans

        vec_bert = self.vectorize(sentences, token_lists, method='BERT')

        print('Clustering ...')
        self.cluster_model = m_clustering(topics.size)
        self.vec[self.method] = self.meta_vectorize(sentences, token_lists, vec_bert)
        self.cluster_model.fit(self.vec[self.method])
        print('Done clustering.')

        print('Fitting ...')
        self.model = Nnclassifier(
            unsup_model = self.cluster_model.predict(self.vec[self.method])
        ).fit(vec_bert, topics)

        print('Done fitting.')

    def predict(self, sentence, token_lists):
        return self.model.predict(
            self.vectorize([sentence], token_lists, method='BERT')
        )

    def score(self, sentences, token_lists, topics):
        predicted_labels = self.model.transduction_
        print(classification_report(topics, predicted_labels))
