# -*- coding: utf-8 -*-
"""Created on Wed May 22 13:20:06 2019@author: milroa1"""
from keras.datasets import imdb

from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM,Embedding
from sklearn.model_selection import train_test_split

import numpy as np

def CreateSimpleLinearDataset(timepoints=100,sz=5):
    X = sum(np.meshgrid(range(sz),range(timepoints)))
    Y = np.array(range(sz,timepoints+sz))
    X = X/timepoints
    Y = Y/timepoints
    Xtrain,Xtest,Ytruetrain,Ytruetest = train_test_split(X,Y,test_size=0.2,random_state=4)
    return Xtrain,Xtest,Ytruetrain,Ytruetest


max_words = 20000
max_sequence_length = 180

(Xtrain,Ytrain),(Xtest,Ytest) = imdb.load_data(num_words=max_words)
words_to_index = imdb.get_word_index()
index_to_word = {v:k for k,v in words_to_index.items() }

from keras.preprocessing import sequence
sequence2 = lambda x: sequence.pad_sequences(x,maxlen=max_sequence_length,padding="post",truncating="post")

Xtrain = sequence2(Xtrain)
Xtest  = sequence2(Xtest )

#############################################

def create_deepLSTMnet(depth=1,hidden_size=32):
    model = Sequential()
    model.add(Embedding(max_words, hidden_size))
    for i in range(depth):
       return_sequences = (depth-1)>i
       model.add(LSTM(hidden_size, activation="tanh",dropout=0.2,recurrent_dropout=0.2,return_sequences=return_sequences))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimezer="adam",metrics=["accuracy"])
    return model
    
model1 = create_deepLSTMnet(1)
model2 = create_deepLSTMnet(2)

epochs = 3

model1.fit(Xtrain,Ytrain,epochs=epochs, shuffel=True)
model1_acc_loss = model1.evaluate(Xtrain,Ytrain)

model2.fit(Xtrain,Ytrain,epochs=epochs, shuffel=True)
model2_acc_loss = model2.evaluate(Xtrain,Ytrain)







import K.prod as Multiply
import K.tanh as Tanh
from keras.layers import Input, Concatenate
class LSTMunit:
    """
            X
            Cin
            Hin
            Cout
            Hout

    """
    
    def __init__(self,sz=12):
        self.sz = sz
        self.build()   
        
    def build(self):
        sz = self.sz
        
        self.Hin_ = Concatenate([Hin, X])
        self.Hin  = Input(shape=(sz,))
        self.Cin  = Input(shape=(sz,))        
        Hin,Cin = self.Hin, self.Cin
        
        self.net1_out = Dense( sz , activation='sigmoid')(Hin)# Forget gate(what to forget in the cell state)
        self.net2_out = Dense( sz , activation='tanh'   )(Hin)# Candidate Layer, net2 and net3 are combined and added to cell state to modify it
        self.net3_out = Dense( sz , activation='sigmoid')(Hin)# Input Gate, net2 and net3 are combined and added to cell state to modify it      
        self.net4_out = Dense( sz , activation='sigmoid')(Hin)# Output Gate, what to output
        
        self.Cout = Multiply([Cin, self.net1_out]) + Multiply([self.net2_out, self.net3_out]) # Memmory State
        self.Hout = Multiply([Tanh(self.Cout), self.net4_out])                           # Hidden State , also Output 


