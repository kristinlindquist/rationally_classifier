import tensorflow as tf
# from tensorflow.distribute.experimental import TPUStrategy
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

class NN_Classifier:
  def __init__(self, latent_dim=64, activation='relu', epochs=100, batch_size=256):
    self.activation = activation
    self.batch_size = batch_size
    self.cc_loss = CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)
    self.epochs = epochs
    self.latent_dim = latent_dim
    self.model = None
    # tf.config.experimental_connect_to_cluster(resolver)
    # tf.tpu.experimental.initialize_tpu_system(resolver)
    # self.strategy = TPUStrategy(resolver)
    self.strategy = MirroredStrategy()
    print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

  def _compile(self, input_dim, num_classes):
    with self.strategy.scope():
      inputs = Input(shape=(input_dim,))
      middle = Dense(self.latent_dim, activation=self.activation)(inputs)
      outputs = Dense(num_classes, activation="softmax")(middle)
      self.model = Model(inputs=inputs, outputs=outputs)

      self.model.compile(
        optimizer=RMSprop(0.01),
        loss=self.cc_loss,
        metrics=['accuracy']
      )

  def fit(self, vec, topics):
    if not self.model:
      self._compile(vec.shape[1], max(topics) + 1)

    labels = tf.keras.utils.to_categorical(topics, max(topics) + 1)
    X_train, X_test, y_train, y_test = train_test_split(vec, labels)

    callbacks = [
      EarlyStopping(monitor='loss', patience=5),
      ModelCheckpoint(filepath='./saved_models/model.{epoch:02d}-{val_loss:.2f}.h5')
    ]

    self.model.fit(
      X_train,
      y_train,
      shuffle=True,
      validation_data=(X_test, y_test),
      batch_size=self.batch_size,
      epochs=self.epochs,
      callbacks=callbacks
    )

  def predict(self, vec):
    return self.model.predict(vec)

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model = keras.models.load_model(filename)
