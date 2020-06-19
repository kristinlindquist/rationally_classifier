from nnclassifier import *
from datetime import datetime
from keras.layers import Dense, Input
from keras.models import Model
import numpy as np
import pandas as pd
from preprocess import *
from sentence_transformers import models, SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import tensorflow as tf
from transformers import *

class Ensemble_Model:
  def __init__(self):
    self.model = None
    self.method = None
    self.bert_model = None
    self.id = datetime.now().strftime("%Y_%m_%d_%H_%M")

  def vectorize(self, text, method=None):
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

      vec = np.array(self.bert_model.encode(text, show_progress_bar=True))
      print('Done BERT vectorizing.')
      return vec

    if method == 'TFIDF':
      print('TFIDF Vectorizing...')
      vectorizer = TfidfVectorizer(min_df=1)
      vec = vectorizer.fit_transform(text)
      print('Done TFIDF Vectorizing.')
      return vec

    else:
      print('Count Vectorizing...')
      vectorizer = CountVectorizer()
      vec = vectorizer.fit_transform(text)
      print('Done Count Vectorizing.')
      return vec

  def _compile(self):
    self.model = NN_Classifier()

  def fit(self, text, topics):
    if not self.model:
      self._compile()

    vec = self.vectorize(text, method='BERT')

    print('Starting fitting.')
    self.model.fit(vec, topics)
    print('Done fitting.')

  def save(self, filename):
    self.model.save("{}.file".format(filename))

  def load(self, filename):
    self.model = NN_Classifier()
    self.model.load(filename)

  def predict(self, sentence):
    prediction = self.model.predict([
      self.vectorize([sentence], method='BERT')
    ])
    return np.argmax(prediction)
