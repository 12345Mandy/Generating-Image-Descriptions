## VGG16-GRU for Generating Image Captions 

By Mandy He [mhe26], Sophia Liu [sliu176], Ria Rajesh [rrajesh]

#### Introduction
We are implementing an existing [paper](http://tamaraberg.com/papers/generation_cvpr11.pdf), and the objective of this paper is to generate text descriptions of images. We chose this paper because it combined many of our interests: computer vision, classification, and natural language processing. Generating image descriptions that closely resemble human speech while also providing positional orientation and descriptors for the objects found in the picture effectively mimic what it’s like to see the image. For people who are visually impaired, having accurate text descriptions of images will allow those people to better understand what is going on in the images. Contrary to previous papers that have focused on retrieving image descriptions from GPS metadata to access relevant text documents, the method used in this paper first detects objects, modifiers (adjectives), and spatial relationships (prepositions) in the image and then uses either an n-gram language model or a simple template-based approach to generate the captions. This method puts more emphasis on the objects and their positions/spatial orientation and attributes. Our implementation, on the other hand, chose a method that combines a CNN with a LSTM to generate captions.

#### Journey
At first, we tried to follow our original paper closely when constructing the model. However, much of the paper was very mathematically focused. This included formulas such as those that determined positions of objects in the images. We found it much too difficult to understand these formulas or how to implement them given our limited knowledge of deep learning methods on images. Hence, we turned to related papers on image caption generation and chose a method that combines a CNN with an LSTM to generate captions (note: the paper that describes this method is linked below).

#### Related Work
We found no public implementations of our original paper. However, we did find implementations of the paper that describe the CNN-LSTM model. In a web article called “How to Develop a Deep Learning Photo Caption Generator from Scratch”, we found [code](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/) related to our topic written in Tensorflow. This article goes through 1) how to obtain the photo and caption dataset, 2) how to prepare the photo data with the VGG16 and pre-compute photo features/embeddings, 3) how to prepare text data through data cleaning, and 4) how to develop the deep learning model. The paper also explains how to evaluate the model with BLEU and how to generate new captions.

#### Datasets
For the image caption generator, we used the Flickr8K dataset, which includes 8092 photographs and files containing descriptions of the photographs. The dataset has a pre-defined training and test dataset which we used to train and test our model. 
  
For training our VGG16 model, we used a Kaggle dataset with images of 10 different animals. The image count varied from 2000-5000 images per class. These images were taken from Google Images.

#### Base, Target, and Stretch Goals
Our base goal was to have our model simply recognize the primary objects in an image, and produce simplified captions that were just sentences describing the primary object in the image. 

Our original target goal would be to have our model recognize primary objects in an image and generate captions that contain modifiers relating to that object besides the word for the object itself. For example, given a picture of a dog, the model should be able to generate a caption like “This is a black dog” or “This is a brown dog”. For our target goal, we intend to base our architecture on the paper linked in the related work section. We ended up changing our target goal to focus more on improving the implementation used in the related work section, by experimenting with different models and architectures. 

Our original stretch goal would be to implement more of the complex functionality seen in the paper, and therefore include the spatial relationships of the objects detected in the images. For example, we would be able to generate a caption like “The cat is on top of the chair”. However, due to the changes we made in the architecture of the project, this stretch goal is no longer relevant. Our model currently takes in a document containing 5 potential captions per image and uses these captions to train our LSTM and hence our merged LSTM-CNN image captioning model. Hence, we chose to make our stretch goal improving the results of the current model.

#### Division of Labor
Our group worked collaboratively on the code and alternated who would be the driver for each meeting. We also divided the code into subgroups. Sophia worked on integrating Inceptionv3 into our code and experimenting with the LSTM/GRU, while Mandy and Ria worked on implementing our own VGG16 model in an attempt to potentially make a better model suited for our purposes of photo content interpretation for caption generation. All three members worked on the writeup and poster and experimented with the model in an attempt to garner better results. Much of the work and discussion was done together over zoom calls and in-person meetings.

#### Methodology
###### VGG16 model
For our final implementation, we based our architecture off of the Stanford paper linked in the related work section. We began by using the VGG16 model implemented in tensorflow. The VGG16 model is a convolutional neural network used for image classification problems.

###### VGG16 Implementation from Scratch
Seeing how half of caption generation is being able to interpret the content from photos, we wanted to see if we could potentially write our own version of VGG16 to classify the main objects in our dataset of images. Despite knowing that the original VGG16 model was trained for weeks, we still wanted to see how we could tweak the model to potentially garner better results. When researching the VGG16 model, we found that most implementations only classified two things, such as cats and dogs. We decided to have 10 classes instead of 2 because 2 would not be sufficient for the image captioning we were doing with our current model that runs on a variety of different images of different objects. 

We were able to implement a VGG16 model that could classify between 10 classes by training it on a Kaggle dataset on 10 types of animals. We split the given dataset into a train and test dataset using the split_folders package. When we tested our implementation of the model we noticed that our loss value dropped significantly while our accuracy never rose above ~0.2, despite our many different attempts to raise the accuracy, such as image data augmentation through Keras’ ImageDataGenerator() function which generates batches of tensor image data with real-time augmentation, our results didn’t get noticeably better. We attempted to use the ImageDataGenerator() function to increase the data sample count through adjusting the rotation range and shifting range arguments of the function, but doing so only made the model run slower without noticeably better accuracy. We considered calling our VGG16 implementation in the image caption generator model but ended up not doing so because this VGG16 was only trained on 10 classes, and the dataset used by the image caption generator to train and test has 1000 classes.

###### Experimentation with GRU and LSTM
We were able to improve our caption generator through experimentation. 

First, we hypothesized that stacking two LSTMs instead of using one LSTM like in the original would create better captions because the two LSTMs would remember longer sequences. We added another LSTM to our original model and got BLEU scores that were within the range of good scores.

Next, to make our model more efficient, we decided to replace the two LSTMs in the caption generator with a GRU since 1) the performance of a GRU is on par with that of LSTMs, and 2) we realized that a GRU would suffice in terms of remembering sequences since our captions are not particularly long.

We changed to using a GRU and got BLEU scores that were just as good, if not better than the ones from using an LSTM or two LSTMs. The model also ran faster with the GRU like our hypothesis suggested. If we had more time, we would run the model several times and time the duration to get more evidence that the GRU increased efficiency.

###### Inceptionv3 Model
We also researched other models for image classification to experiment with, including ResNet50, Inceptionv3, and EfficientNet. We found that Inceptionv3 was ranked as the best model for image classification, so we attempted to make our implementation better by using the Inceptionv3 model instead of the VGG16 model. Moreover, while VGG16 outputs 4096 features/dimensions, Inceptionv3 has a dimension space of 2048. This means that the feature vectors in VGG16 are likely more sparse; many of the values will be close to 0. Hence, our hypothesis was that Inceptionv3 may perform better when it comes to feature engineering. We tried to alter our implementation to use the Inceptionv3 model rather than VGG16 to improve our results, but we were unable to integrate the Inceptionv3 model because the output shapes of the two models were incompatible. We debugged our code by studying the architecture of the two models, the output shapes, and the target image sizes, as suggested by our mentor TA, but were ultimately unable to fix the error and run our model with Inceptionv3. However, our prediction is that this model would have performed significantly better as it is better suited for feature engineering. 

#### Challenges
The main challenge we faced was reconciling the differences between the implementation of image caption generation in the original paper and the image caption generator we could realistically implement in code. The original paper deals with certain architectures that were beyond the scope of what we had learned in class, so we chose to make selective simplifications in the finalized version of the project by following a related paper. The integration of different sources made designing our model more difficult, because we needed to continuously analyze the differences between their implementations and ours. 

Another challenge we encountered was storing and moving large files, specifically, features.pkl, which was produced in preprocessing. Since features.pkl was over the size limit for pushing to git, we learned how to use git-lfs (large file storage) to move the file into the remote repository and resolved the git issue. 

Furthermore, experimenting with different parts of the model was challenging since the article that we referenced used a lot of Keras functions that we had never used before. For instance, because of the CNN LSTM architecture that we were using, the data we fed into training was immense. Not only did the embeddings of each image need to be copied 5 times for each of the 5 captions corresponding to the image in the descriptions file holding all the captions, we also needed to copy each of the images another n times where n is the number of words in each caption. For reference, the number of training images is 6000. Due to this large amount of data, we used Keras model.fit_generator rather than the usual methods of training we used in class (see [notes on fit_generator](https://www.geeksforgeeks.org/keras-fit-and-keras-fit_generator/)). However, this function felt like a black-box and even after trying to familiarize ourselves with such libraries and functions, we constantly ran into errors that happened whilst calling fit_generator due to mismatches in size in the process of evaluating this function which made debugging incredibly difficult. This is the main error we ran into when trying to integrate the pretrained Inceptionv3 model into our code.

#### Results
###### VGG16 from Scratch Implementation Results
To test the success of our VGG16 implementation, we printed the accuracy and change in loss at each step of the training of the model. The final accuracy ranged from values between 0.19-0.2.

###### Experimentation with the Image Caption Generator: BLEU Scores
We used BLEU scores to determine the success of our models. We tested our models with the following criteria, which we got from “Where to put the Image in an Image Caption Generator”.

*Criteria*

**Ranges of “good” BLEU scores at each stage:**
BLEU-1: 0.401 to 0.578
BLEU-2: 0.176 to 0.390
BLEU-3: 0.099 to 0.260
BLEU-4: 0.059 to 0.170

*Our Testing Results*

The following scores use the pre-implemented VGG16 model and various caption generators.

**Scores for VGG16-LSTM**
BLEU-1: 0.518058
BLEU-2: 0.280059
BLEU-3: 0.186328
BLEU-4: 0.082154

**Scores for VGG16-GRU**
BLEU-1: 0.555404
BLEU-2: 0.297118
BLEU-3: 0.200846
BLEU-4: 0.092077

**Scores for VGG16-LSTM-LSTM**
BLEU-1: 0.556409
BLEU-2: 0.296571
BLEU-3: 0.187125
BLEU-4: 0.073948

These scores all generally fall into the ranges of “good” BLEU scores.