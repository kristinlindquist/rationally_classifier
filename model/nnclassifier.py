import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from sklearn.model_selection import train_test_split

class NN_Classifier:
  def __init__(self, latent_dim=32, activation='relu', epochs=200, batch_size=128):
    self.activation = activation
    self.batch_size = batch_size
    self.cc_loss = CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)
    self.epochs = epochs
    self.latent_dim = latent_dim
    self.model = None

  def _compile(self, input_dim, num_classes):
    inputs = Input(shape=(input_dim,))
    middle = Dense(self.latent_dim, activation=self.activation)(inputs)
    outputs = Dense(num_classes, activation="softmax")(middle)
    self.model = Model(inputs=inputs, outputs=outputs)

    self.model.compile(
      optimizer=keras.optimizers.RMSprop(0.01),
      loss=self.cc_loss,
      metrics=['accuracy']
    )

  def fit(self, vec, topics):
    if not self.model:
      self._compile(vec.shape[1], max(topics) + 1)

    labels = keras.utils.to_categorical(topics, max(topics) + 1)
    X_train, X_test, y_train, y_test = train_test_split(vec, labels)

    callback = EarlyStopping(monitor='loss', patience=3)

    self.model.fit(
      X_train,
      y_train,
      shuffle=True,
      validation_data=(X_test, y_test),
      batch_size=self.batch_size,
      epochs=self.epochs,
      callbacks=[callback]
    )

  def predict(self, vec):
    return self.model.predict(vec)

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model = keras.models.load_model(filename)
