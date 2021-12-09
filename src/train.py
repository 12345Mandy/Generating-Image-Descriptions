import numpy as np
from numpy import array
from pickle import load
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

#TODO: 
#1. make epoch a constant here?
#2. remove tokenizer completely from here
#3. potentially change ifs to opposite
# shift train to main and make this its own class.




# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions
 
# load photo features
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
 
# covert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc
 
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
	lines = to_lines(descriptions)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# calculate the length of the description with the most words
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)
 
# create sequences of images, input sequences and output words for an image
# def create_sequences(tokenizer, max_length, desc_list, photo, vocab_size):
def create_sequences(max_length, desc_list, photo, vocab_size, word_to_index):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		# encode the sequence

		seq = [word_to_index[w] for w in desc.split() if w in word_to_index]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			# pad input sequence
			# print(type(in_seq))
			# print(type(max_length))
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
 
# define the captioning model #change all this to be how we do it 
def define_model(vocab_size, max_length):
	# feature extractor model
	inputs1 = Input(shape=(4096,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,))
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = GRU(256)(se2)
	# se3 = LSTM(256, return_sequences=True)(se2)
	# se4 = LSTM(256)(se3)

	# decoder model
	# decoder1 = add([fe2, se4])
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu')(decoder1)
	outputs = Dense(vocab_size, activation='softmax')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	# plot_model(model, to_file='model.png', show_shapes=True)
	return model
 
# data generator, intended to be used in a call to model.fit_generator()
# def data_generator(descriptions, photos, tokenizer, max_length, vocab_size):
def data_generator(descriptions, photos, max_length, vocab_size, word_to_index):
	# loop for ever over images
	while 1:
		for key, desc_list in descriptions.items():
			# retrieve the photo feature
			photo = photos[key][0]
			# in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
			in_img, in_seq, out_word = create_sequences(max_length, desc_list, photo, vocab_size, word_to_index)
			yield [in_img, in_seq], out_word
 
# def train(model, train_descriptions, train_features, tokenizer, vocab_size, max_length): #maybe add batching?
# def train(model, train_descriptions, train_features, vocab_size, max_length, word_to_index): 
# 	steps = len(train_descriptions) # train the model, run epochs manually and save after each epoch
# 	epochs = 1
# 	for i in range(epochs):
# 		# create the data generator
# 		# generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
# 		generator = data_generator(train_descriptions, train_features, max_length, vocab_size, word_to_index)
# 		# fit for one epoch
# 		model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
# 		# save model
# 		model.save('model_' + str(i) + '.h5')


