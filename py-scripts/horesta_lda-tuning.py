#!/usr/bin/env python
# coding: utf-8


# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
# https://medium.com/@kurtsenol21/topic-modeling-lda-mallet-implementation-in-python-part-1-c493a5297ad2
# https://jeriwieringa.com/projects/dissertation/
# https://radimrehurek.com/gensim/models/wrappers/ldamallet.html

# http://journalofdigitalhumanities.org/2-1/words-alone-by-benjamin-m-schmidt/
# https://mimno.infosci.cornell.edu/topics.html
# https://tedunderwood.com/2012/04/07/topic-modeling-made-just-simple-enough/


# Packages

import os
import pandas as pd
import json
from datetime import datetime as dt
import re
import itertools
from itertools import compress
from itertools import chain
import ast
import spacy


# Plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={'figure.figsize':(20,12)})


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.wrappers import LdaMallet
from gensim.test.utils import get_tmpfile, common_texts
from gensim.corpora import MalletCorpus

nlp = spacy.load("da_core_news_sm", disable=['parser', 'ner'])

data_path = os.path.join('D:/', 'data', 'horesta') # Remember to update
out_path = os.path.join('..', 'output')


# Loading data
path = os.path.join(data_path, 'horesta_posts_2021-03-25.json')

with open(path, 'r') as file:
    data = json.load(file)
    

# Tokenizer

def tokenizer_spacy(text, stop_words=list(nlp.Defaults.stop_words), tags=['NOUN', 'ADJ', 'VERB', 'ADV']):
       
    text = text.replace('\n', ' ')
    numbers_re = r".*\d.*"
    punct_regex = r"[^\w\s]"
    
    doc = nlp(text)
        
    pos_tags = tags # Keeps proper nouns, adjectives and nouns
    
    exceptions = ['covid', 'corona']
    
    tokens = []
      
    for word in doc:
        if ((word.pos_ in pos_tags) or (any([exception in word.text for exception in exceptions]))) and (len(word.lemma_) > 4) and (word.lemma_.lower() not in stop_words) and not (re.match(numbers_re, word.lemma_.lower())):
            token = word.lemma_.lower() # Returning the word in lower-case.
            token = re.sub(punct_regex, "", token)
            tokens.append(token)

    return(tokens)


# Tokenize data

for entry in data:
    entry['tokens'] = tokenizer_spacy(entry.get('text'))


# Dictionary and filter extremes
id2token = corpora.Dictionary([entry.get('tokens') for entry in data])

#id2token.filter_extremes(no_below=0.05, no_above=0.65)


# Gensim doc2bow corpus

for entry in data:
    entry['doc2bow'] = id2token.doc2bow(entry.get('tokens'))    
    
tokens_bow = [entry.get('doc2bow') for entry in data]


# Tfidf weighting of doc2bow 

tfidf = gensim.models.TfidfModel(tokens_bow)

for entry in data:
    entry['tfidfbow'] = tfidf[entry.get('doc2bow')]
    
tokens_tfidf = [entry.get('tfidfbow') for entry in data]
corpus_texts = [entry.get('tokens') for entry in data]


# Parameters for tuning

topic_nums = list(range(2,30))
chunksizes = list(range(500,2000, 100))
passes = list(range(10,100, 10))
iterations = list(range(500,5000, 1000))

para_combinations = [combination for combination in itertools.product(topic_nums, chunksizes, passes, iterations)]

parameters_combinations = []
for combination in para_combinations:
    parameter_combination = {}
    parameter_combination['num_topics'] = combination[0]
    parameter_combination['chunksize'] = combination[1]
    parameter_combination['passes'] = combination[2]
    parameter_combination['iterations'] = combination[3]
    
    parameters_combinations.append(parameter_combination)


# Computing coherence for different parameters

coherence_scores = []

for parameters in parameters_combinations:
    lda_model = gensim.models.LdaModel(corpus = tokens_tfidf, 
                                           num_topics = parameters.get('num_topics'), 
                                           id2word = id2token, 
                                           chunksize = parameters.get('chunksize'), 
                                           passes = parameters.get('passes'), 
                                           iterations = parameters.get('iterations'), 
                                           random_state = 1332,
                                           alpha = "auto",
                                           eta = "auto")
    
    coherence_model_lda = CoherenceModel(model=lda_model, corpus=tokens_tfidf, coherence='u_mass')
    coherence_scores.append(coherence_model_lda.get_coherence())


# Export coherence scores

filename_out = "coherence_scores.txt"

with open(os.path.join(out_path, filename_out), 'w', encoding = 'utf-8') as f:
    for line in coherence_scores:
        f.write(line + "\n")

