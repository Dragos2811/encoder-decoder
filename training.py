'''
Created on Aug 10, 2018

@author: Dragos2811
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.models import load_model
from keras.layers.wrappers import Bidirectional
from reader import create
import numpy as np
import random


n_cell1 = 124
n_cell2 = 64
n_epoch = 3
seq_lengh = 64
num_samples = 75000
word_to_ix,ix_to_word,vocab_size,data=create(1,3000)

def no_future_end(n):
        for i in range(n,n+seq_lengh):
            if data[i] == '/html':
                return i+3
        return 0
def generate(data,num_samples):
    X=np.zeros((num_samples,int(seq_lengh/10),vocab_size))
    y=np.zeros((num_samples,int(seq_lengh/10*9), vocab_size)) #out_seq_lenght=1
    n = 0
    
    for i in range(num_samples):
        if n+seq_lengh >= len(data):
            n = random.randint(0,int(len(data)/2))
        r = no_future_end(n)
        if r != 0:
            n = r
        for j in range (int(seq_lengh/10)):
            
            X[i][j][word_to_ix[data[n]]] = 1
            n += 1
        for j in range(int(seq_lengh/10*9)):
            y[i][j][word_to_ix[data[n]]] = 1
            n += 1
    return X,y





try:
    model = load_model('./tmp/s%se%sseq%s%sx%s.h5' % (num_samples,n_epoch,seq_lengh,n_cell1,n_cell2))
    print("Model loaded")
except ValueError:
    print("No model with given hyperparameters found. Creating a new one")
    model = Sequential()
    model.add(Bidirectional(LSTM(n_cell1, input_shape=(int(seq_lengh/10), vocab_size))))
    model.add(RepeatVector(int(seq_lengh/10*9))) #out_seq_lenght
    model.add(Bidirectional(LSTM(n_cell2, return_sequences=True)))
    model.add(TimeDistributed(Dense(vocab_size, activation= 'softmax' )))
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    
    #fit LSTM
    X,y=generate(data,num_samples)
    model.fit(X, y, epochs=n_epoch, batch_size = 64)
    
    # evaluate LSTM
    _,_,_,data=create(3001,4000)
    X,y=generate(data,num_samples)
    loss, acc = model.evaluate(X, y, verbose=0)
    print( 'Loss: %f, Accuracy: %f' % (loss, acc*100))
    
    #saving model
    model.save('./tmp/s%se%sseq%s%sx%s.h5' % (num_samples,n_epoch,seq_lengh,n_cell1,n_cell2))
    print("model saved")

# evaluate LSTM
_,_,_,data=create(3001,4000)
X,y=generate(data,num_samples)
loss, acc = model.evaluate(X, y, verbose=0)
print( 'Loss: %f, Accuracy: %f' % (loss, acc*100))
       
#generate html
X,_=generate(data[int(seq_lengh/10):],1)
y = model.predict(X, verbose=0)
print("\n************\n")
for i in range(int(seq_lengh/10)):
    print(data[i],end="")
for i in y[0]:
    print(ix_to_word[np.argmax(i)],end="")
print("\n************\nGenerating done")