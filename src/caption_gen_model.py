import numpy as np
from numpy import array
from pickle import load
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.models import Model
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.utils.vis_utils import plot_model 
from tensorflow.keras.models import load_model
from utils import *
import constant



class Caption_Gen:
	def __init__(self, vocab_size, max_description_length):
		# The model class inherits from tf.keras.Model.
        # It stores the trainable weights as attributes.
		self.vocab_size = vocab_size
		self.max_length = max_description_length

	def get_model_GRU(self):
		# feature extractor model
		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)
		# sequence model
		inputs2 = Input(shape=(self.max_length,))
		se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = GRU(256)(se2)

		# decoder model
		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
		# tie it together [image, seq] [word]
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		# compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# summarize model
		model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)
		return model
	
	def get_model_LSTM_double(self):
		# feature extractor model
		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)
		# sequence model
		inputs2 = Input(shape=(self.max_length,))
		se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = LSTM(256, return_sequences=True)(se2)
		se4 = LSTM(256)(se3)

		# decoder model
		decoder1 = add([fe2, se4])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
		# tie it together [image, seq] [word]
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		# compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# summarize model
		model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)
		return model

	def get_model_LSTM(self):
		# feature extractor model
		inputs1 = Input(shape=(4096,))
		fe1 = Dropout(0.5)(inputs1)
		fe2 = Dense(256, activation='relu')(fe1)
		# sequence model
		inputs2 = Input(shape=(self.max_length,))
		se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
		se2 = Dropout(0.5)(se1)
		se3 = LSTM(256)(se2)
		# decoder model
		decoder1 = add([fe2, se3])
		decoder2 = Dense(256, activation='relu')(decoder1)
		outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
		# tie it together [image, seq] [word]
		model = Model(inputs=[inputs1, inputs2], outputs=outputs)
		# compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# summarize model
		model.summary()
		# plot_model(model, to_file='model.png', show_shapes=True)
		return model




