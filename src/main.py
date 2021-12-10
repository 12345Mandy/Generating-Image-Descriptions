from pickle import dump
from nltk.translate.bleu_score import corpus_bleu
from caption_gen_model import Caption_Gen

from preprocess import *
from utils import *
from constant import *
import matplotlib.pyplot as plt

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

def train(model, train_descriptions, train_features, vocab_size, max_length, word_to_index): 
	steps = len(train_descriptions) # train the model, run epochs manually and save after each epoch
	for i in range(constant.EPOCH):
		# create the data generator
		generator = data_generator(train_descriptions, train_features, max_length, vocab_size, word_to_index)
		# fit for one epoch
		model.fit_generator(generator, epochs=constant.EPOCH, steps_per_epoch=steps, verbose=1)
		# save model
		model.save('model_' + str(i) + '.h5')

# test the model using bleu
def bleu(model, descriptions, photos, max_length, indextoword, wordtoindex):
    actual, predicted = list(), list()
    acc = 0
    for key, list_of_descr in descriptions.items():
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
    index_to_word, word_to_index = own_tokenizer(vocab)
    vocab_size = len(index_to_word) + 1
    print('Vocabulary Size: %d' % vocab_size)
    lst_of_desc = list()
    for key in train_descriptions.keys():
        [lst_of_desc.append(d) for d in train_descriptions[key]]
    #lines = to_lines(train_descriptions)
    max_description_length = max(len(d.split()) for d in lst_of_desc)
    print('Description Length: %d' % max_description_length)


    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #define the model-> this is our caption generator. 
    
    model = Caption_Gen(vocab_size, max_description_length)
    model = model.get_model_GRU() ## can be replaceed with the other get_models on caption_gen_models
    train(model, train_descriptions, train_features, vocab_size, max_description_length, word_to_index)

    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # test
    print("loading descriptions and features for test")
    test_descriptions, test_features = load_descriptions_and_features(TEST_IMAGES_FILENAME, "test")
    
    # load the model that has just been trained
    filename = MODEL_FILENAME
    print("loading model")
    model = load_model(filename)
    # print("ready to test bleu")
    # bleu(model, test_descriptions, test_features, max_description_length, index_to_word, word_to_index)

    # OUTPUT_DIM = 4069
    for z in range(10): # set higher to see more examples
        pic = list(test_features.keys())[z]
        image = test_features[pic]#.reshape((1, OUTPUT_DIM))
        x = plt.imread(os.path.join('../data/Flicker8k_Dataset', pic) + '.jpg')
        plt.imshow(x)
        plt.show()
        caption = generate_caption(model, image, max_description_length, index_to_word, word_to_index)
        caption = caption.replace('startseq', '')
        caption = caption.replace('endseq', '')
        caption = caption.strip()
        print("Caption:", caption)
        print("_____________________________________")
    

if __name__ == '__main__':
    main()
 