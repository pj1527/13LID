import os
import librosa
import numpy as np
import sys
import pickle

data_folder = './13_language_dataset/'
num_langs = len([name for name in os.listdir(data_folder) if not name.startswith('.')])

l=0
data={}
labels={}
for lang in os.listdir(data_folder):
    f=1
    if not lang.startswith('.'):
        for wavfile in os.listdir(data_folder+lang):
            num_files = len([name for name in os.listdir(data_folder+lang) if not name.startswith('.')])
            sig, fs = librosa.load(data_folder+lang+'/'+wavfile)
            S = librosa.feature.melspectrogram(y=sig, sr=fs)

            if (f%50) is 1: #To print status after every 50 generated spectrograms
                print('Generating spectrograms for {} ({}/{} wavfiles done). {} languages remaining'.format(lang,f,num_files,num_langs-l-1))

            if f is 1:
                data[l]=[]
                labels[l]=[]
            data[l].append(S)
            labels[l].append(l)

            f=f+1
        l=l+1

class_names = []
for lang in os.listdir(data_folder):
    if not lang.startswith('.'):
        class_names.append(lang)

class_names = np.asarray(class_names)
print('Languages to be classified and available in the dataset : {}'.format(class_names))

for l in range(class_names.shape[0]):
    if l is 0:
        corpus = np.asarray(data[l])
        targets = np.asarray(labels[l])
    else:
        corpus = np.vstack((corpus,data[l]))
        targets = np.hstack((targets,labels[l]))

print('Dataset dimensions :')
print('Number of files = {}'.format(corpus.shape[0]))
print('Dimensions of each file = {}'.format(corpus[0].shape))
print('Targets = {}'.format(targets.shape[0]))

corpus = np.reshape(corpus,(corpus.shape[0],corpus.shape[1],corpus.shape[2],1))
print(corpus.shape)

with open('13L_mel_spectrograms.pickle','wb') as f:
    pickle.dump((corpus,targets),f)
