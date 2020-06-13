from nnclassifier import *
from unsup_ensemble import *
from datetime import datetime
from preprocess import *
from sentence_transformers import models, SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import *
from keras.layers import Average, Dense, Input
from keras.models import Model

class Ensemble_Model:
    def __init__(self):
        self.model = None
        self.method = None
        self.id = datetime.now().strftime("%Y_%m_%d_%H_%M")

    def vectorize(self, sentences, token_lists, method=None):
        if method is None:
            method = self.method

        if method == 'TFIDF':
            print('TF-IDF ...')
            tf_idf = TfidfVectorizer()
            vec = tf_idf.fit(sentences)
            print('TF-IDF done.')
            return vec

        elif method == 'COUNT':
            print('Count vectorizing ...')
            cv = CountVectorizer()
            vec = cv.fit_transform(sentences)
            print('Count vectorizing done.')
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

    def ensemble(models, inputs):
        outputs = [model.outputs[0] for model in models]
        y = Average()(outputs)

        return Model(inputs = inputs, outputs = y, name='ensemble')

    def _compile(self, vec1, vec2, topics):
        sup_input = Input(shape=(vec1.shape[1],))
        unsup_input_1 = Input(shape=(vec2.shape[1],))
        unsup_input_2 = Input(shape=(vec1.shape[1],))

        sup_nn = Nnclassifier(sup_input)
        sup_nn.fit(vec1, topics)

        unsup2x = unsup_ensemble(unsup_input_1, unsup_input_2)
        unsup2x.fit(vec2, vec1, topics)

        self.model = self.ensemble(
            [sup_nn.model, unsup2x],
            [sup_input, unsup_input_1, unsup_input_2]
        )

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.KLDivergence(),
            metrics=['accuracy']
        )

        self.model.summary()

    def fit(self, sentences, token_lists, topics):
        vec_bert = self.vectorize(sentences, token_lists, method='BERT')
        vec_tfidf = self.vectorize(sentences, token_lists, method='COUNT')

        if not self.model:
            self._compile(vec_bert, vec_tfidf, topics)

        self.model.fit(x = [vec_bert, vec_tfidf, vec_bert], y = keras.utils.to_categorical(topics, max(topics) + 1))
        print('Done fitting.')

    def predict(self, sentence, token_lists):
        return self.model.predict(
            self.vectorize([sentence], token_lists, method='BERT')
        )
