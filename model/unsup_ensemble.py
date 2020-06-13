from nnclassifier import *
from keras.layers import Average, Dense
from keras.models import Model

class Unsup_Ensemble:
    def __init__(self, input1, input2, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.input1 = input1
        self.input2 = input2

    def ensemble(models, inputs):
        outputs = [model.outputs[0] for model in models]
        y = Average()(outputs)

        return Model(inputs = inputs, outputs = y, name='ensemble')

    def _compile(self, vec1, vec2, topics):
        unsup_nn1 = Nnclassifier(input1)
        unsup_nn2 = Nnclassifier(input2)
        unsup_nn1.fit(vec_tfidf, topics)
        unsup_nn2.fit(vec_bert, topics)

        self.model = ensemble(
            [unsup_nn1.model, unsup_nn2.model],
            [input1, input2]
        )

        self.model.compile(
            optimizer='adam',
            loss=keras.losses.KLDivergence(),
            metrics=['accuracy']
        )

    def fit(self, vec1, vec2, labels):
        topics = keras.utils.to_categorical(labels, max(labels) + 1)

        if not self.model:
            self._compile(vec1, vec2, topics)

        self.model.fit(
            x = [vec1, vec2],
            y = labels,
            batch_size=self.batch_size,
            epochs=self.epochs
        )

    def predict(self, vec):
        return self.model.predict(vec, batch_size=vec.size)