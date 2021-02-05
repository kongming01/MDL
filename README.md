# MDL
This repository contains the data for the paper "Detecting Product Adoption Intention via Multi-View Deep Learning." The code will be available soon.

# Datasets
In the above paper, the datasets "iphone", "foreo", and "movie" are used in the experiments. In each dataset folder, there are two types of files for each dataset.

The file "tweets_no_urls.txt" contains the preprocessed tweets where one tweet is at one row. 
The file "labels.txt" includes the labels for the tweets in the file "tweets_no_urls.txt". In the file "labels.txt", the first column is the index of a tweet, and the second column is the corresponding label.

In addition, the pre-trained tweet embeddings of 200 dimensions (downloaded from https://nlp.stanford.edu/projects/glove/) are used in the embedding layer of our proposed model.
