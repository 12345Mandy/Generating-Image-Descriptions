import numpy as np
from numpy import array
from pickle import load
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.models import Model
from keras_preprocessing.text import Tokenizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model
from utils import *
import constant
from constant import *



 
def own_tokenizer(vocab):
    indextoword = {}
    wordtoindex = {}
    idx = 1
    for word in vocab:
        wordtoindex[word] = idx
        indextoword[idx] = word
        idx += 1
    return indextoword, wordtoindex

# generate image description
def generate_caption(model, photo, max_length, indextoword, wordtoindex):
    desc = STARTSEQ
    for _ in range(max_length):
        # integer encode input sequence
        seq = [wordtoindex[w] for w in desc.split() if w in wordtoindex]
        seq = pad_sequences([seq], maxlen=max_length)
        # predict next word
        next_word = model.predict([photo, seq], verbose=0)
        # convert probability to integer
        next_word = np.argmax(next_word)
        word = indextoword[next_word]
        if word is None:
            break
        desc = desc + ' ' + word
        if word == ENDSEQ:
            break
    return desc

# input = result from generate_caption
def clean_up_caption(desc):
    desc = desc.split()[1:-1]
    desc = ' '.join(desc)
    return desc

# loads the previously saved photo embeddings and descriptions
def load_descriptions_and_features(filename, train_or_test):
	# load dataset (6K) -> this is training or test depending on parameter
	filename = filename
	dataset = load_set(filename)
	print('Dataset', train_or_test, ': = %d' % len(dataset))
	# get the clean descriptions from descriptions.txt
	doc = load_doc('descriptions.txt')
	descriptions = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		image_id, image_desc = tokens[0], tokens[1:]
		# Only use images in the set
		if image_id in dataset:
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	print('Descriptions', train_or_test, ': = %d' % len(descriptions))
	# photo features->load all photo features
	all_features = load(open('features.pkl', 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	print('Photos', train_or_test, ': = %d' % len(features))
	return descriptions, features



 # load doc into memory
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	for line in doc.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)


# create sequences of images, input sequences and output words for an image
def create_input_sequences(max_length, desc_list, photo, vocab_size, word_to_index):
	X1, X2, y = list(), list(), list()
	# walk through each description for the image
	for desc in desc_list:
		seq = [word_to_index[w] for w in desc.split() if w in word_to_index]
		# split one sequence into multiple X,y pairs
		for i in range(1, len(seq)):
			# split into input and output pair
			in_seq, out_seq = seq[:i], seq[i]
			in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
			# encode output sequence
			out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
			# store
			X1.append(photo)
			X2.append(in_seq)
			y.append(out_seq)
	return array(X1), array(X2), array(y)
 
 
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, photos, max_length, vocab_size, word_to_index):
	while 1:
		for key, desc_list in descriptions.items():
			photo = photos[key][0]
			in_img, in_seq, out_word = create_input_sequences(max_length, desc_list, photo, vocab_size, word_to_index)
			yield [in_img, in_seq], out_word
