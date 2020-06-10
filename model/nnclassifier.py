import keras
from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split

class Nnclassifier:
    def __init__(self, unsup_model, latent_dim=32, activation='relu', epochs=100, batch_size=128):
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.model = None
        self.cc_loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        self.unsup_model = unsup_model
        self.step = 0

    def _compile(self, input_dim, num_classes):
        inputs = Input(shape=(input_dim,))
        # self.model.add(Activation(self.activation, name="activation"))
        predictions = Dense(num_classes, name="hidden")(inputs)
        predictions = Activation('softmax', name="softmax")(predictions)
        self.model = Model(inputs = inputs, outputs = predictions)

        self.model.summary()

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(0.01),
            loss=self.custom_loss(),
            metrics=['accuracy']
        )

    def custom_loss(self):
        def loss(y_true, y_pred):
            start_idx = self.step * self.batch_size
            end_idx = start_idx + self.batch_size
            unsup_results = self.unsup_model[start_idx : end_idx,]
            self.step += 1
            self.step %= self.batch_size
            return (
                K.mean(K.square(y_pred - y_true), axis=-1)
                * self.cc_loss(y_true, y_pred)
                * unsup_results
            )
        return loss

    def fit(self, data, sup_labels):
        if not self.model:
            self._compile(data.shape[1], max(sup_labels) + 1)

        self.model.fit(
            data,
            keras.utils.to_categorical(sup_labels, max(sup_labels) + 1),
            batch_size=self.batch_size,
            epochs=self.epochs
        )

