# import tensorflow
# import keras
# from os import listdir
from pickle import dump#, load
from nltk.translate.bleu_score import corpus_bleu
from caption_gen_model import Caption_Gen
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.models import Model
# import string
# from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from array import array

from preprocess import *
# from train import *
from utils import *
from constant import *

# TRAIN_IMAGES_FILENAME = '../data/Flickr8k_text/Flickr_8k.trainImages.txt'
# TEST_IMAGES_FILENAME = '../data/Flickr8k_text/Flickr_8k.testImages.txt'
# MODEL_FILENAME = 'model_0.h5' ## this should be created after running train
# STARTSEQ = 'startseq'
# ENDSEQ = 'endseq'
# THRESH = 10

# TODO: potentially shift all functions besides train, bleu and main to utils

def load_descriptions_and_features(filename, train_or_test):
    # load training dataset (6K)
    filename = filename
    dataset = load_set(filename)
    print('Dataset', train_or_test, ': = %d' % len(dataset))
    # descriptions
    descriptions = load_clean_descriptions('descriptions.txt', dataset)
    print('Descriptions', train_or_test, ': = %d' % len(descriptions))
    # photo features
    features = load_photo_features('features.pkl', dataset)
    print('Photos', train_or_test, ': = %d' % len(features))
    return descriptions, features



# filename = TEST_IMAGES_FILENAME
#     test = load_set(filename)
#     print('Dataset: %d' % len(test))
#     # descriptions
#     test_descriptions = load_clean_descriptions('descriptions.txt', test)
#     print('Descriptions: test=%d' % len(test_descriptions))
#     # photo features
#     test_features = load_photo_features('features.pkl', test)
#     print('Photos: test=%d' % len(test_features))

# map an integer to a word
# def word_given_id(looking_for, tokenizer):
# 	for word, index in tokenizer.word_index.items():
# 		if index == looking_for:
# 			return word
# 	return None

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
# def generate_caption(model, tokenizer, photo, max_length):
def generate_caption(model, photo, max_length, indextoword, wordtoindex):
    desc = STARTSEQ
    for _ in range(max_length):
        # integer encode input sequence
        # seq = tokenizer.texts_to_sequences([desc])[0]
        seq = [wordtoindex[w] for w in desc.split() if w in wordtoindex]
        seq = pad_sequences([seq], maxlen=max_length)
        # predict next word
        next_word = model.predict([photo, seq], verbose=0)
        # convert probability to integer
        next_word = np.argmax(next_word)
        # word = word_given_id(next_word, tokenizer)
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

def train(model, train_descriptions, train_features, vocab_size, max_length, word_to_index): 
	steps = len(train_descriptions) # train the model, run epochs manually and save after each epoch
	for i in range(constant.EPOCH):
		# create the data generator
		# generator = data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
		generator = data_generator(train_descriptions, train_features, max_length, vocab_size, word_to_index)
		# fit for one epoch
		model.fit_generator(generator, epochs=constant.EPOCH, steps_per_epoch=steps, verbose=1)
		# save model
		model.save('model_' + str(i) + '.h5')

# test the model using bleu
# def bleu(model, descriptions, photos, tokenizer, max_length):
def bleu(model, descriptions, photos, max_length, indextoword, wordtoindex):
    actual, predicted = list(), list()
    acc = 0
    for key, list_of_descr in descriptions.items():
        # generated = generate_caption(model, tokenizer, photos[key], max_length)
        print(key, acc)
        generated = generate_caption(model, photos[key], max_length, indextoword, wordtoindex)
        generated = generated.split()
        references = [desc.split() for desc in list_of_descr]
        actual.append(references)
        predicted.append(generated)
        acc = acc + 1
        
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


def main():
    preprocess_load_all()
    # load training dataset (6K)
    train_descriptions, train_features = load_descriptions_and_features(TRAIN_IMAGES_FILENAME, "train")

    # get captions for training
    train_captions = []
    for _, val in train_descriptions.items():
        for caption in val:
            train_captions.append(caption)

    # remove words that occur infrequently
    word_counts = {}
    num_sentences = 0
    for sentence in train_captions:
        num_sentences += 1
        sentence = sentence.split(' ')
        for word in sentence:
            word_counts[word] = word_counts.get(word, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= THRESH]
    # tokenizer = create_tokenizer(train_descriptions)
    index_to_word, word_to_index = own_tokenizer(vocab)
    vocab_size = len(index_to_word) + 1 
    # vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    lines = to_lines(train_descriptions)
    max_description_length = max(len(d.split()) for d in lines)
    #max_description_length = max_length(train_descriptions)
    print('Description Length: %d' % max_description_length)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # define the model-> TIP: this is our caption generator with the lstm stuff
    
    #model = define_model(vocab_size, max_description_length) 
    model = Caption_Gen(vocab_size, max_description_length)
    model = model.get_model_GRU()

    # train
    # train(model, train_descriptions, train_features, tokenizer, vocab_size, max_description_length)
    
    train(model, train_descriptions, train_features, vocab_size, max_description_length, word_to_index)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # test
    print("loading descriptions and features for test")
    test_descriptions, test_features = load_descriptions_and_features(TEST_IMAGES_FILENAME, "test")
    
    # load the model that has just been trained
    filename = MODEL_FILENAME
    print("loading model")
    model = load_model(filename)
    # evaluate model
    # bleu(model, test_descriptions, test_features, tokenizer, max_description_length)
    print("ready to test bleu")
    bleu(model, test_descriptions, test_features, max_description_length, index_to_word, word_to_index)

if __name__ == '__main__':
    main()
 






    # # right now main assumes preprocess has been run
    # preprocess_load_all()
    # # load training dataset (6K)
    # filename = '../data/Flickr8k_text/Flickr_8k.trainImages.txt'
    # train_stuff = load_set(filename)
    # print('Dataset: %d' % len(train_stuff))
    # # descriptions
    # train_descriptions = load_clean_descriptions('descriptions.txt', train_stuff)
    # print('Descriptions: train=%d' % len(train_descriptions))
    # # photo features
    # train_features = load_photo_features('features.pkl', train_stuff)
    # print('Photos: train=%d' % len(train_features))

    # # prepare tokenizer
    # tokenizer = create_tokenizer(train_descriptions)
    # vocab_size = len(tokenizer.word_index) + 1
    # print('Vocabulary Size: %d' % vocab_size)

    # max_length = max_length(train_descriptions)
    # print('Description Length: %d' % max_length)

    # ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = define_model(vocab_size, max_length) # define the model

    # # train
    # train(model, train_descriptions, train_features, tokenizer, vocab_size, max_length)

    # ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # # load test set
    # filename = '../data/Flickr8k_text/Flickr_8k.testImages.txt'
    # test = load_set(filename)
    # print('Dataset: %d' % len(test))
    # # descriptions
    # test_descriptions = load_clean_descriptions('descriptions.txt', test)
    # print('Descriptions: test=%d' % len(test_descriptions))
    # # photo features
    # test_features = load_photo_features('features.pkl', test)
    # print('Photos: test=%d' % len(test_features))
    
    # # # load the model
    # filename = 'model_0.h5'
    # model = load_model(filename)
    # # evaluate model
    # evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)