#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2016 Anurag Roy <anu15roy@gmail.com>
#
# Distributed under terms of the MIT license.

# coding: utf-8


import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time
from CIKM_stemmer import Word2Vec_stemmer


print "Running the stemmer"

''' Creating an Word2Vec_stemmer object runs the stemming algorithm '''

k = Word2Vec_stemmer('nepal_model_for_stem', 'wv_glove_twitter.txt', beta = 0.9, gamma=0.7, prefix=2, lambda_val=3)




''' following command shows stem and the corresponding unstemmed list of words '''
d = k.stem_dict
f = open('word_stems_list.txt' ,'w')
for key, val in d.iteritems():
    f.write(key+" "+str(val)+"\n")
f.close()

''' to get stem of a word'''
for word in k.model.wv.vocab.keys():
    print "stem of {} is {}".format(word, k.get_stem(word))
