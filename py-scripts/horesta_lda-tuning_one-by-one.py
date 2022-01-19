#!/usr/bin/env python
# coding: utf-8

# Packages

import os
import pandas as pd
import json
import re
import itertools
from itertools import compress
from itertools import chain
import spacy


# Gensim

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

nlp = spacy.load("da_core_news_sm", disable=['parser', 'ner'])

data_path = os.path.join('..', 'data') # Remember to update
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
topic_nums = [2,3,4,7]
chunksizes = [2000]
passes = list(range(20,101, 20))
iterations = list(range(500,2001, 500))

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
filename_out = "coherence_parameter-tuning.jsonl"

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
    
    coherence_score = coherence_model_lda.get_coherence()
    
    coherence_score_out = dict()
    coherence_score_out['num_topics'] = parameters.get('num_topics')
    coherence_score_out['chunksize'] = parameters.get('chunksize')
    coherence_score_out['passes'] = parameters.get('passes')
    coherence_score_out['iterations'] = parameters.get('iterations')
    coherence_score_out['coherence_u_mass'] = coherence_score
    
    coherence_scores.append(coherence_score_out)
    
    # Export coherence scores
    if not os.path.isfile(os.path.join(out_path, filename_out)):
        with open(os.path.join(out_path, filename_out), 'w', encoding = 'utf-8') as f:
            f.write(str(coherence_score_out) + "\n")
    else:
        with open(os.path.join(out_path, filename_out), 'a', encoding = 'utf-8') as f:
            f.write(str(coherence_score_out) + "\n")
            
            
    print(f"Calculated {len(coherence_scores)}/{len(parameters_combinations)} parameter possibilities... \n", end = "\r")