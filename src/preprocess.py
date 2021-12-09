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
 

 
# extract descriptions for images
def load_descriptions(doc): 
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# remove filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping
 
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

def create_clean_descriptions_map(unclean_desc_doc):
	id_desc_map = dict()
	# process lines
	for line in unclean_desc_doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, and remove filename from it
		image_id = tok[0].split('.')[0]
		# rest is description
		image_desc = tok[1:]

		# Cleanup description
		# convert to lower case
		image_desc = [word.lower() for word in desc]
		# remove punctuation from each token
		image_desc = [w.translate(table) for w in desc]
		# remove hanging 's' and 'a'
		image_desc = [word for word in desc if len(word)>1]
		# remove tokens with numbers in them
		image_desc = [word for word in desc if word.isalpha()]
		if image_id not in id_desc_map:
			id_desc_map[image_id] = list()
		id_desc_map[image_id].append(' '.join(image_desc))
      
     


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

 
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc
 

 ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~new functions~~~~~~~~~~~~~~~~~~
# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()


def save_photo_features():
	# extract features from all images
	directory = '../data/Flicker8k_Dataset'
	features = extract_photo_features(directory)
	print('Extracted Features: %d' % len(features))
	# save to file
	dump(features, open('features.pkl', 'wb')) #this creates features file
	

def save_clean_descriptions():
	file = '../data/Flickr8k_text/Flickr8k.token.txt' #these are like the labels, pregenerated captions
	# load descriptions
	doc = load_doc(file)
	# parse descriptions
	descriptions = load_descriptions(doc)
	print('Loaded: %d ' % len(descriptions))
	# clean descriptions
	clean_descriptions(descriptions)
	# summarize vocabulary
	vocabulary = to_vocabulary(descriptions)
	print('Vocabulary Size: %d' % len(vocabulary))
	# save to file
	save_descriptions(descriptions, 'descriptions.txt') #these are the labels, what we train based off of (modify to remove prepositions and stuff)



def preprocess_load_all():
	'''
    Loads image features/embeddings and cleaned descriptions into
	features.pkl and descriptions.txt respectively 
	ONLY IF THESE FILES DO NOT EXIST
	otherwise will do nothing
    
    :return: None
    '''
	photo_features_path = 'features.pkl'   # assuming u are running src
	descriptions_path = 'descriptions.txt'   # assuming u are running src
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

	


	
# # extract features from all images
# directory = '../data/Flicker8k_Dataset'
# features = extract_photo_features(directory)
# print('Extracted Features: %d' % len(features))
# # save to file
# dump(features, open('features.pkl', 'wb')) #this creates features file

# file = '../data/Flickr8k_text/Flickr8k.token.txt' #these are like the labels, pregenerated captions
# # load descriptions
# doc = load_doc(file)
# # parse descriptions
# descriptions = load_descriptions(doc)
# print('Loaded: %d ' % len(descriptions))
# # clean descriptions
# clean_descriptions(descriptions)
# # summarize vocabulary
# vocabulary = to_vocabulary(descriptions)
# print('Vocabulary Size: %d' % len(vocabulary))
# # save to file
# save_descriptions(descriptions, 'descriptions.txt') #these are the labels, what we train based off of (modify to remove prepositions and stuff)