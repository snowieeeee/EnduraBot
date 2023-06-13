import json

import nltk
import pickle
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import numpy as np
import pandas as pd
import random
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers.legacy import SGD


with open('data.json', 'r') as file:
    data_file = file.read()
    intents = json.loads(data_file)
    
words=[]
classes = []
documents = []
ignore_words = ['?', '!', ',', '.']
for intent in intents['intents']:
  for pattern in intent['patterns']:
    #tokenize each word
    w = nltk.word_tokenize(pattern)
    #add to our words
    words.extend(w)
    #add to documents 
    documents.append((w,intent['tag']))
    #add to our classes list
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
      
words =[lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes=sorted(set(classes))
# documents = combination between patterns and intents
print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
pickle.dump(words,open('texts.pkl','wb'))
pickle.dump(classes,open('labels.pkl','wb'))
#create my training data
training=[]
#create an empty array for the output
output_empty=[0]*len(classes)

#training set : bag of words for each sentence
for doc in documents:
  bag=[]
  #list of tokenized words for the pattern
  pattern_words = doc[0]
  #lemmatizing each word
  pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
  #create my bag of words array with 1 if word match found in current position
  for word in words:
    bag.append(1) if word in pattern_words else bag.append(0)

  #output is 0 for each tag and 1 for current tag (for each pattern)
  output_row=list(output_empty)
  output_row[classes.index(doc[1])]=1
  
  training.append([bag,output_row])



training=np.array(training)
training = training.tolist()
training.append([bag, output_row])

# shuffle the features and turn them into numpy array
random.shuffle(training)
training = np.array(training)

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])
train_y = list(training[:, 1])

print("Training data created")



model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu', name="layer1"))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', name="layer2"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax', name="layer3"))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('endu_model.h5', hist)
print("model created")
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


