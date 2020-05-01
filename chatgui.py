import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from collections import defaultdict

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random


words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list
    

    
    

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result
    
def getRespons(ints, intents_json):
    tag = ints.split(',')[0]
   
    list_of_intents = intents_json['intents']
    resul=''
    for i in list_of_intents:
        if(i['tag']== tag):
            resul = random.choice(i['responses'])
            break
    return resul
    


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
    



    
def chatbot_test(questions,model,words):
    intest_line = questions.split('\n')
    return_list1 = []
    answers=[]
    for t in intest_line:
      p1=bow(t, words, show_details=False)
      
      res1 = model.predict(np.array([p1]))[0]
      ERROR_THRESHOLD = 0.25
      results = [[i,r] for i,r in enumerate(res1) if r>ERROR_THRESHOLD]
      # sort by strength of probability
      results.sort(key=lambda x: x[1], reverse=True)
      
      
      for r in results:
        return_list1.append(classes[r[0]]+','+'probability'+ str(r[1]))
        
    for resp in return_list1:
      
      answers.append(getRespons(resp, intents))
    #print(answers)
     
      


    
      
      
    return answers
      
def process(questions,model,words,test_ans):
    
    respons = chatbot_test(questions,model,words)
    
    ans_line = test_ans.split('\n')
    count=0
    total=(len(ans_line))
    #print(len(ans_line))
    #print(len(respons))
    
    for i in range(len(respons)):

      
      if respons[i] == ans_line[i]:
        count=count+1
    print(count)
    Accuracy = count/total
    print('The Accuracy of the model is: ',Accuracy*100) 
    
    
    
    
      
    
      


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    
    
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
  
intents = json.loads(open('intents.json').read())
questions = open('questions.txt').read()
test_ans = open('answers.txt').read()
process(questions,model,words,test_ans)
base = Tk()
base.title("Ask a Question")
base.geometry("600x700")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="500", width="500", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height="8",
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="40", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=575,y=6, height=570)
ChatLog.place(x=6,y=6, height=570, width=585)
EntryBox.place(x=150, y=601, height=90, width=390)
SendButton.place(x=6, y=601, height=90)

base.mainloop()

