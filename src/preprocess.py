import tensorflow as tf
import os
from os import listdir
from pickle import dump
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import string
from utils import load_doc
from constant import *
 



def create_clean_descriptions_map(unclean_desc_doc):
	id_desc_map = dict()
	# translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	# process lines
	for line in unclean_desc_doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, and remove filename from it
		image_id = tokens[0].split('.')[0]
		# rest is description
		image_desc = tokens[1:]

		# Cleanup description
		# convert to lower case
		image_desc = [word.lower() for word in image_desc]
		# remove punctuation from each token
		image_desc = [w.translate(table) for w in image_desc]
		# remove hanging 's' and 'a'
		image_desc = [word for word in image_desc if len(word)>1]
		# remove tokens with numbers in them
		image_desc = [word for word in image_desc if word.isalpha()]
		if image_id not in id_desc_map:
			id_desc_map[image_id] = list()
		id_desc_map[image_id].append(' '.join(image_desc))
	return id_desc_map
      
     


def extract_photo_features(directory): #this needs to stay 
	'''
	extract features from each photo in the directory
	output: imageid as keys and features as values of a dictionary

	taken from https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
	note: we really tried to rewrite this function to handle other types of 
	models both pretrained and self trained/implemented. However, as stated in our 
	writeup, it was quite the challenge and so we reverted back to the pretrained vgg16
	that the article used
	'''
	# load the model
	model = tf.keras.applications.vgg16.VGG16()
	# re-structure the model
	model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
		image = load_img(filename, target_size=(224, 224))
		# convert the image pixels to a numpy array
		image = img_to_array(image)
		# reshape data for the model
		image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
		# prepare the image for the VGG model
		image = preprocess_input(image)
		# get features
		feature = model.predict(image, verbose=0)
		# get image id
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

 



def save_photo_features():
	# extract features from all images
	directory = PHOTOS_DIRECTORY
	features = extract_photo_features(directory)
	print('Extracted Features: %d' % len(features))
	# save to file
	dump(features, open('features.pkl', 'wb')) #this creates features file
	

def save_clean_descriptions():
	file = DESCRIPTIONS_UNCLEAN_FILENAME #these are like the labels, pregenerated captions
	# load descriptions
	doc = load_doc(file)
	#load descriptions and  clean descriptions
	descriptions = create_clean_descriptions_map(doc)
	print('Loaded: %d ' % len(descriptions))
	# summarize vocabulary and save description mapping to descriptions.txt
	all_desc_vocab = set()
	for key in descriptions.keys():
		[all_desc_vocab.update(d.split()) for d in descriptions[key]]
	print('Vocabulary Size: %d' % len(all_desc_vocab))
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open('descriptions.txt', 'w')
	file.write(data)
	file.close()
	



def preprocess_load_all():
	'''
    Loads image features/embeddings and cleaned descriptions into
	features.pkl and descriptions.txt respectively 
	ONLY IF THESE FILES DO NOT EXIST
	otherwise will do nothing
    
    :return: None
    '''
	photo_features_path = 'features.pkl'   # assuming u are running from src
	descriptions_path = 'descriptions.txt'   # assuming u are running from src
	isFeatureExtracted = os.path.exists(photo_features_path)
	existsDescriptions = os.path.exists(descriptions_path)
	if not isFeatureExtracted:
		print("CREATING features.pkl")
		save_photo_features()
	print("SAVED features.pkl")
	if not existsDescriptions:
		print("CREATING descriptions.txt")
		save_clean_descriptions()
	print("SAVED descriptions.txt")

	