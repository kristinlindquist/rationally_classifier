import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split

class Autoencoder:
    def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
        self.latent_dim = latent_dim
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.history = None

    def _compile(self, input_dim):
        input_vec = Input(shape=(input_dim,))
        encoded = Dense(self.latent_dim, activation=self.activation)(input_vec)
        decoded = Dense(input_dim, activation=self.activation)(encoded)
        self.encoder = Model(input_vec, encoded)

        self.autoencoder = Model(input_vec, decoded)
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.mean_squared_error)

        encoded_input = Input(shape=(self.latent_dim,))
        self.decoder = Model(encoded_input, self.autoencoder.layers[-1](encoded_input))

    def fit(self, X):
        if not self.autoencoder:
            self._compile(X.shape[1])
        X_train, X_test = train_test_split(X)
        self.history = self.autoencoder.fit(
            X_train,
            X_train,
            epochs=200,
            batch_size=128,
            shuffle=True,
            validation_data=(X_test, X_test),
            verbose=0
        )
