# CIKM2017
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-2.7-blue.svg)](https://www.python.org/)


## Table of Contents

* [Summary](#summary)
* [Dependencies](#dependencies)
* [Hyperparameters and Options](#hyperparameters-and-options)
* [Data for Demo](#data-for-demo)
* [Run Demo](#run-demo)

### Summary
Implementation of the proposed algorithm in the paper **Combining Local and Global Word Embeddings for Microblog Stemming** by Anurag Roy, Trishnendu Ghorai, Kripabandhu Ghosh, Saptarshi Ghosh. The proposed unsupervised algorithm finds stems of all words using help of local and global word embeddings.

### Dependencies
python version: `python 2.7`

packages: 
- `gensim`
- `nltk`
- `scikit_learn`

To install the dependencies run `pip install -r requirements.txt`

### Hyperparameters and Options
Hyperparameters and options in `unsupclean.py`.

- `model_file` gensim Word2Vec model trained on the corpus
- `global_model_file` text file of the global wordvectors in googles word2vec format
- `alpha` The alpha value used in the algorithm  \[0, 1\]
- `beta` The beta value used in the algorithm  \[0, 1\]
- `prefix` The prefix length of words matched
- `m` The minimum length of strings considered
- `lambda_val` The lambda value used in the algorithm  \[0, 1\]
### Data for Demo
For the local word embedding, we have trained a word2vec model on tweets of Nepal Earthquake from the FIRE 2016 microblog track collection, and for the Global word embeddings we provide the pretained GloVe embeddings of words from [twitter dataset](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip). We changed the format of the GloVe embeddings to word2vec format. The datasets can be accessed from the following sources:

1. [Word2Vec Embeddings](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/EZeKmDyb2G5PqPJpzu0crPcBhHuhudOsmM51V8h5Tle68g?e=ojJMMZ)
2. [Global Embeddings](https://iitkgpacin-my.sharepoint.com/:u:/g/personal/anurag_roy_iitkgp_ac_in/ERwTmZwmvnFDih7YSY9hk2ABOo-eWk-fU65huVAOQHMxSQ?e=JE06kw)

The downloaded zipped data files should be extracted in the CIKM2017 folder.

### Run Demo

To generate the list of word stems, run the following command:

`python2 driver.py`

The word stem list will be stored in the `word_stems_list.txt` file. Each line in the file contains:
```
<stem> <list of words to be replaced with the stem>
```

An example entry of the file will be 
```
msghelpea [u'msghelpea', u'msghelpeart', u'msghelpearthqu']
```


