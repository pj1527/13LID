# 13LID
This repository contains a CNN architecture to classify 13 Indian Languages from their spoken utterance. 2 second short utterances have also been classified.

This is not a very complex model but still performs good, gives accuracy of around 70% for 5 second utterances and 66.7% for 2 second ones. Performance through heatmaps (which helps in identifying confused language pairs) is also shown below.

The dataset used in this repo is described in this [paper](https://www.semanticscholar.org/paper/An-Investigation-of-Deep-Neural-Network-for-in-MounikaK.-Achanta/5f6ffd39e74a66492cfb34b62a21e91d08332e35).

## Requirements
1. python3
2. librosa - to generate mel spectrograms
3. numpy
4. sklearn
5. tensorflow-gpu 1.10.0
6. keras - 2.2.4

## Hardware
1 Nvidia GeForce GTX 1080 Ti gpu was used for training (gpu recommended), which was available through my institute's high performance computing cluster and it uses slurm as workload manager.

## How to use it
1. Make a folder for your dataset, inside which all languages have their separate folders containing their respective wav files. Then change the value of data_folder variable in all feature extraction codes.
2. Then run feature extraction to generate mel spectrograms.
3. Then use the classifier notebook to train the CNN models. This notebook saves your trained model in hdf5 format.
4. You can classify.py script to classify any utterance by typing the following command :
python3 classify.py path_file,
where path_file is the path to the file you want to classify.

## Theory
mel spectrogram = spectrogram with frequency mapped to mel space. [See](https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#melspectrogram)

Now since mel spectrograms are a 2-dimensional signal, we can use CNNs as we do for image classification.

I have used a combination of convolution+pooling+dropout twice, then 2 dense layers followed by softmax activation. All activations are done with Relu. For detailed description regarding the model, please look at the model summary given in the classification notebook.

## Heatmaps
We can clearly see that gujarati-marathi is the most confused language pair for our classifier :
![alt text](https://github.com/pj1527/13LID/blob/master/2sec_heatmap.png "Heatmap for 2sec utterances")
