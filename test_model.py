import nltk
from nltk.tokenize import word_tokenize
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
from matplotlib import pyplot

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

def train():
    elements=[]
    classes = []
    eleTagPair = []
    ignore_charecters = ['?', '!']
    data_file = open('tri_train.json').read()
    intents = json.loads(data_file)


    for intent in intents['intents']:
        for Ques in intent['patterns']:

        #tokenize each word
            w = nltk.word_tokenize(Ques)
            elements.extend(w)
        #add documents in the corpus
            eleTagPair.append((w, intent['tag']))

        # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
    elements = [lemmatizer.lemmatize(w.lower()) for w in elements if w not in ignore_charecters]
    elements = sorted(list(set(elements)))
# sort classes
    classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
    #print (len(eleTagPair), "eleTagPair")
    # classes = intents
    #print (len(classes), "classes", classes)
# words = all words, vocabulary
    #print (len(elements), "unique lemmatized elements", elements)


    pickle.dump(elements,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
    trainData = []
# create an empty array for our output
    output_empty = [0] * len(classes)
# training set, bag of words for each sentence
    for doc in eleTagPair:
    # initialize our bag of words
        bag = []
    # list of tokenized words for the pattern
        pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
        for w in elements:
            bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
    
        trainData.append([bag, output_row])
#print(training)
# shuffle our features and turn into np.array
    random.shuffle(trainData)
    trainData = np.array(trainData)
# create train and test lists. X - patterns, Y - intents
    train_x = list(trainData[:,0])
#print(train_x)
    train_y = list(trainData[:,1])
    print("Training data created")
    return train_x, train_y,elements

def test(train_word_len):
    elements=[]
    classes = []
    eleTagPair = []
    ignore_charecters = ['?', '!']
    data_file = open('tri_test.json').read()
    intents = json.loads(data_file)


    for intent in intents['intents']:
        for Ques in intent['patterns']:

        #tokenize each word
            w = nltk.word_tokenize(Ques)
            elements.extend(w)
        #add documents in the corpus
            eleTagPair.append((w, intent['tag']))

        # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
    elements = [lemmatizer.lemmatize(w.lower()) for w in elements if w not in ignore_charecters]
    elements = sorted(list(set(elements)))
    diff = train_word_len - len(elements)
    #print(diff)
    pad(elements,'x',train_word_len)
    #print(len(words))
# sort classes
    classes = sorted(list(set(classes)))
# eleTagPair = combination between patterns and intents
    #print (len(documents), "documents")
    # classes = intents
    #print (len(classes), "classes", classes)
# elements = all words, vocabulary
    #print (len(elements), "unique lemmatized words", elements)


    pickle.dump(elements,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))

# create our training data
    testData = []
# create an empty array for our output
    output_empty = [0] * len(classes)
# training set, bag of words for each sentence
    for doc in eleTagPair:
    # initialize our bag of words
        bag = []
    # list of tokenized words for the pattern
        pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
        for w in elements:
            bag.append(1) if w in pattern_words else bag.append(0)
    
    # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
    
        testData.append([bag, output_row])
#print(training)
# shuffle our features and turn into np.array
    random.shuffle(testData)
    testData = np.array(testData)
# create train and test lists. X - patterns, Y - intents
    test_x = list(testData[:,0])
#print(train_x)
    test_y = list(testData[:,1])
    print("Testing data created")
    return test_x, test_y

def pad(l, content, width):
     l.extend([content] * (width - len(l)))


train_x, train_y,train_words = train()
test_x,test_y = test(len(train_words))
    
    
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
history = model.fit(np.array(train_x), np.array(train_y),validation_data=(np.array(test_x), np.array(test_y)), epochs=200, batch_size=5, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(np.array(train_x), np.array(train_y), verbose=0)
_, test_acc = model.evaluate(np.array(test_x), np.array(test_y), verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

