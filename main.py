import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from preparedata import PrepareAudios

x = pd.read_csv("WordsMatrix.csv").values
y = PrepareAudios("MainDataset/male").fit()

input_shape = (x.shape[1],)
output_shape = (y.shape[1], y.shape[2])

print("input shape: ", input_shape)
print("output shape: ", output_shape)


class ReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, target_shape, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.target_shape = target_shape

    def call(self, inputs):
        return tf.reshape(inputs, self.target_shape)

def create_text_to_audio_model(text_shape, audio_shape):
  text_input = Input(shape=text_shape, name='text_input')
  audio_input = Input(shape=audio_shape, name='audio_input')
  text_lstm = LSTM(units=128, return_sequences=True)(text_input)
  text_lstm = Dropout(0.2)(text_lstm)
  text_lstm = LSTM(units=64)(text_lstm)
  audio_lstm = LSTM(units=128, return_sequences=True)(audio_input)
  audio_lstm = Dropout(0.2)(audio_lstm)
  audio_lstm = LSTM(units=64)(audio_lstm)
  merged = Concatenate()([text_lstm, audio_lstm])
  output = Dense(units=audio_shape[0] * audio_shape[1], activation='linear')(merged)
  reshape_layer = ReshapeLayer(audio_shape)
  output = reshape_layer(output)
  model = Model(inputs=[text_input, audio_input], outputs=output)

  return model

# Example usage:
text_shape = (67,)
audio_shape = (128, 165)
model = create_text_to_audio_model(text_shape, audio_shape)
model.compile(loss='mse', optimizer='adam')
model.summary()

X_train, X_test, y_train, y_test = x[:(x.shape[0] * 80 // 100)], x[(x.shape[0] * 80 // 100):], y[:(y.shape[0] * 80 // 100)], y[(y.shape[0] * 80 // 100):]
history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=100,
    validation_data=(X_train, y_test),
)
