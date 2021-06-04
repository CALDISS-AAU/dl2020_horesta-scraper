#!/usr/bin/env python
# coding: utf-8

# Packages

import os
import json
import re
import spacy

# Gensim
import gensim
import gensim.corpora as corpora

nlp = spacy.load("da_core_news_sm", disable=['parser', 'ner'])

data_path = os.path.join('/work', 'dl2020_horesta-scraper', 'data')
datawork_path = os.path.join('/work', 'dl2020_horesta-scraper', 'data', 'work')
out_path = os.path.join('/work', 'dl2020_horesta-scraper', 'output')

filename_in = 'horesta_posts_2021-03-25.json'
filename_out = 'horesta_posts_2021-03-25_tokenized.json'


# Loading data
path = os.path.join(data_path, filename_in)

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
        if len(word.lemma_) <= 4:
            continue
        if word.lemma_.lower() in stop_words:
            continue
        if re.match(numbers_re, word.lemma_.lower()):
            continue
        if ((word.pos_ in pos_tags) or (any([exception in word.text for exception in exceptions]))):
            token = word.lemma_.lower() # Returning the word in lower-case.
            token = re.sub(punct_regex, "", token)
            tokens.append(token)

    return(tokens)


# Tokenize data

for entry in data:
    entry['tokens'] = tokenizer_spacy(entry.get('text'))
    
    
# Dictionary

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
    

# Write data with tokens

with open(os.path.join(datawork_path, filename_out), 'w', encoding = 'utf-8') as f:
    json.dump(data, f)