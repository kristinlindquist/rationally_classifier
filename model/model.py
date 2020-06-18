from nnclassifier import *
from datetime import datetime
from keras.layers import Average, Dense, Input
from keras.models import Model
import pandas as pd
from preprocess import *
from sentence_transformers import models, SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import *

class Ensemble_Model:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.model = None
        self.method = None
        self.bert_model = None
        self.id = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def vectorize(self, sentences, method=None):
        if method is None:
            method = self.method

        if method == 'BERT':
            print('Vectorizing for BERT ...')

            if self.bert_model is None:
                word_embedding_model = models.Transformer('emilyalsentzer/Bio_ClinicalBERT')

                pooling_model = models.Pooling(
                    word_embedding_model.get_word_embedding_dimension(),
                    pooling_mode_mean_tokens=True,
                    pooling_mode_cls_token=False,
                    pooling_mode_max_tokens=False
                )
                self.bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            vec = np.array(self.bert_model.encode(sentences, show_progress_bar=True))
            print('Done BERT vectorizing.')
            return vec

        if method == 'TFIDF':
            print('TFIDF Vectorizing...')
            vectorizer = TfidfVectorizer(min_df=1)
            vec = vectorizer.fit_transform(sentences)
            print('Done TFIDF Vectorizing.')
            return vec

        else:
            print('Count Vectorizing...')
            vectorizer = CountVectorizer()
            vec = vectorizer.fit_transform(sentences)
            print('Done Count Vectorizing.')
            return vec            

    def ensemble(self, models, inputs):
        outputs = [model.outputs[0] for model in models]
        y = Average()(outputs)

        return Model(inputs = inputs, outputs = y, name='ensemble')

    def _compile(self, vec1, vec2, topics):
        oname_input = Input(shape=(vec1.shape[1],))
        mname_input = Input(shape=(vec2.shape[1],))

        self.model1 = Nnclassifier(oname_input)
        self.model1.fit(vec1, topics)

        self.model2 = Nnclassifier(mname_input)
        self.model2.fit(vec2, topics)

        self.model = self.ensemble(
            [self.model1.model, self.model2.model],
            [oname_input, mname_input]
        )

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.KLDivergence(),
            metrics=['accuracy']
        )

        self.model.summary()

    def fit(self, sentences, topics, m_names):
        vec_oname_bert = self.vectorize(sentences, method='BERT')
        vec_mname_bert = self.vectorize(m_names, method='BERT')

        if not self.model:
            self._compile(vec_oname_bert, vec_mname_bert, topics)

        self.model.fit(x = [vec_oname_bert, vec_mname_bert], y = keras.utils.to_categorical(topics, max(topics) + 1))
        print('Done fitting.')

    def save(self, filename):
        self.model.save("{}.file".format(filename))
        self.model1.save("{}-1.file".format(filename))
        self.model2.save("{}-2.file".format(filename))

    def load(self, filename):
        self.model = Nnclassifier()
        self.model.load(filename)

    def predict(self, sentence, m_name):
        prediction = self.model.predict([
            self.vectorize([sentence], method='BERT'),
            self.vectorize([m_name], method='BERT')
        ])
        return np.argmax(prediction)
