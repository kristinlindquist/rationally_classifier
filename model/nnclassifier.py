import keras
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.model_selection import train_test_split

class Nnclassifier:
    def __init__(self, inputs=None, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.model = None
        self.inputs = inputs
        self.cc_loss = keras.losses.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)

    def _compile(self, input_dim, num_classes):
        inputs = Input(shape=(input_dim,)) if self.inputs is None else self.inputs
        predictions = Dense(self.latent_dim, activation=self.activation)(inputs)
        predictions = Dense(num_classes, activation="softmax")(predictions)
        self.model = Model(inputs = inputs, outputs = predictions)

        self.model.compile(
            optimizer=keras.optimizers.RMSprop(0.01),
            loss=self.cc_loss,
            metrics=['accuracy']
        )

    def fit(self, data, labels):
        if not self.model:
            self._compile(data.shape[1], max(labels) + 1)

        cat_labels = keras.utils.to_categorical(labels, max(labels) + 1)

        X_train, X_test, y_train, y_test = train_test_split(data, cat_labels)

        self.model.fit(
            X_train,
            y_train,
            shuffle=True,
            validation_data=(X_test, y_test),
            batch_size=self.batch_size,
            epochs=self.epochs
        )

    def predict(self, vec):
        return self.model.predict(vec)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = keras.models.load_model(filename)
