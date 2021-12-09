# import tensorflow
# import keras
# from os import listdir
from pickle import dump#, load
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
from train import *
from utils import *

TRAIN_IMAGES_FILENAME = '../data/Flickr8k_text/Flickr_8k.trainImages.txt'
TEST_IMAGES_FILENAME = '../data/Flickr8k_text/Flickr_8k.testImages.txt'
MODEL_FILENAME = 'model_0.h5' ## this should be created after running train

# def load_descriptions_and_features():




def main():
    # right now main assumes preprocess has been run
    preprocess_load_all()
    # load training dataset (6K)
    filename = TRAIN_IMAGES_FILENAME
    train_stuff = load_set(filename)
    print('Dataset: %d' % len(train_stuff))
    # descriptions
    train_descriptions = load_clean_descriptions('descriptions.txt', train_stuff)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    train_features = load_photo_features('features.pkl', train_stuff)
    print('Photos: train=%d' % len(train_features))

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    max_description_length = max_length(train_descriptions)
    print('Description Length: %d' % max_description_length)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # model = define_model(vocab_size, max_description_length) # define the model

    # # train
    # train(model, train_descriptions, train_features, tokenizer, vocab_size, max_description_length)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # load test set
    filename = TEST_IMAGES_FILENAME
    test = load_set(filename)
    print('Dataset: %d' % len(test))
    # descriptions
    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    # photo features
    test_features = load_photo_features('features.pkl', test)
    print('Photos: test=%d' % len(test_features))
    
    # # load the model
    filename = 'model_0.h5'
    model = load_model(filename)
    # evaluate model
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_description_length)

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