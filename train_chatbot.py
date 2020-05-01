import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

elements=[]
classes = []
eleTagPair = []
ignore_charecters = ['?', '!']
data_file = open('Train.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for Ques in intent['patterns']:

        #tokenize each word
        w = nltk.word_tokenize(Ques)
        elements.extend(w)
        #add eleTagPair in the corpus
        eleTagPair.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
elements = [lemmatizer.lemmatize(w.lower()) for w in elements if w not in ignore_charecters]
elements = sorted(list(set(elements)))
# sort classes
classes = sorted(list(set(classes)))
# eleTagPair = combination between patterns and intents
print (len(eleTagPair), "eleTagPair")
# classes = intents
print (len(classes), "classes", classes)
# elements = all elements, vocabulary
print (len(elements), "unique lemmatized elements", elements)


pickle.dump(elements,open('elements.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# create our trainData data
trainData = []
# create an empty array for our output
output_empty = [0] * len(classes)
# trainData set, bag of elements for each sentence
for doc in eleTagPair:
    # initialize our bag of elements
    bag = []
    # list of tokenized elements for the Ques
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related elements
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of elements array with 1, if word match found in current Ques
    for w in elements:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each Ques)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    trainData.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(trainData)
trainData = np.array(trainData)
# create train and test lists. X - patterns, Y - intents
train_x = list(trainData[:,0])
train_y = list(trainData[:,1])
print("Train data is created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('Model.h5', hist)

print("Model Created")
