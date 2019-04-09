import os
import librosa
import librosa.display
import numpy as np
import sys
import pickle
from keras.models import load_model

sig, fs = librosa.load(sys.argv[1],sr=None)
            
a = 0.97
emp_sig = np.append(sig[0],sig[1:] - a*sig[:-1]) #emphasized signal y(t)=x(t)-a*x(t-1)

win_length = int(round(0.025*fs)) # for 25 ms frame length
hop_length = int(round(0.01*fs)) # for 10ms shift between frames
d = librosa.stft(emp_sig,n_fft=512,win_length=win_length,hop_length=hop_length)
D = np.abs(d)**2 #to get power spectrum
S = librosa.feature.melspectrogram(S=D)
S = np.reshape(S,(1,S.shape[0],S.shape[1],1))
#corpus = np.reshape(corpus,(corpus.shape[0],corpus.shape[1],corpus.shape[2],1))

with open('class_names.pickle','rb') as f:
    class_names = pickle.load(f)
    
model = load_model('13LID_CNN.h5')
pred = model.predict(S)
pred = np.argmax(pred, axis=1)

print('This utterance is of {} language'.format(class_names[pred]))
