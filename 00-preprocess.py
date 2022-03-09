

# Import packages
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import spacy
import gensim
from gensim.parsing.preprocessing import strip_numeric
import requests
import csv

# Download data from Risky Data, hosted on Google Cloud
print('Downloading data.')
DATA_URL = 'https://storage.googleapis.com/risky-data-open-source/ipo.csv'
ipo_df = pd.read_csv(DATA_URL)
ipo_df.to_csv('data/ipo.csv', index=False)
print('Data successfully downloaded.')

# Drop IPOs without a risk factors section
ipo_df = ipo_df.dropna(subset=['RF_clean_paragraphs'])

# Define sample for Risky Data paper
ipo_df.reset_index(inplace=True, drop=True)

print('Full sample shape: {}'.format(ipo_df.shape))

# Split data into paragraphs
ipo_all = pd.concat([pd.Series(idx, row['RF_clean_paragraphs'].split('---new_paragraph---')) for idx, row in ipo_df.iterrows()]).reset_index()
ipo_all.columns = ['RF', 'id']
pars_all_raw = ipo_all['RF'].values

print('All pars: {}'.format(ipo_all.shape))

# Pre-processing
def cleanData(pars):
    stemmer = SnowballStemmer(language='english')

    # -----------------
    # 1) lowercase, tokenize and remove punctuation/digits
    # -----------------

    pars_tokenized = []

    for par in pars:

        if type(par) == float:
            par = ''

        par = gensim.utils.simple_preprocess(par, deacc=True, min_len=2)
        par = [strip_numeric(word) for word in par]

        pars_tokenized.append(par)

    assert len(pars) == len(pars_tokenized), 'Data lost: tokenization'
    print('\tTokenization done.')

    # -----------------
    # 2) remove stop words
    # -----------------

    pars_stop = []

    stop_words = stopwords.words('english')
    stop_words.extend(['may', 'might', 'could'])

    for par in pars_tokenized:
        pars_stop.append([word for word in par if word not in stop_words])

    assert len(pars) == len(pars_stop), 'Data lost'
    print('\tRemoving stop words done.')

    # -----------------
    # 3) keep nouns/adjectives/verbs/adverbs only
    # -----------------

    pars_pos = []

    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    allowed_postags = ['NOUN', 'PROPN', 'VERB']

    for par in pars_stop:
        # POS filtering
        par = nlp(" ".join(par))
        par = [token.text for token in par if token.pos_ in allowed_postags]

        # remove stop words
        par = [word for word in par if word not in stop_words]
        pars_pos.append(par)

    assert len(pars) == len(pars_pos), 'Data lost'
    print('\tPOS cleaning done.')

    # -----------------
    # 4) stem words
    # -----------------

    pars_stems = []

    for par in pars_pos:
        # lemmatize
        par = [stemmer.stem(token) for token in par]
        pars_stems.append(par)

    assert len(pars) == len(pars_stems), 'Data lost'
    print('\tStemming done.')

    # summary stats of how words change
    all_tokens = [s for d in pars_tokenized for s in d]
    print('\n\tSummary stats')
    print("\tNumber of total tokens = {}".format(len(all_tokens)))
    print("\tNumber of unique tokens = {}\n".format(len(set(all_tokens))))

    all_stops = [s for d in pars_stop for s in d]
    print("\tNumber of tokens without stop words = {}".format(len(all_stops)))
    print("\tNumber of unique tokens without stop words = {}\n".format(len(set(all_stops))))

    all_pos = [s for d in pars_pos for s in d]
    print("\tNumber of tokens after POS cleaning = {}".format(len(all_pos)))
    print("\tNumber of unique tokens after POS cleaning = {}\n".format(len(set(all_pos))))

    all_stems = [s for d in pars_stems for s in d]
    print("\tNumber of stems = {}".format(len(all_stems)))
    print("\tNumber of unique stems = {}\n".format(len(set(all_stems))))

    return pars_stems

# Clean text data
print('\nCleaning all paragraphs.')
pars_all = cleanData(pars_all_raw)
ipo_all['RF_clean_prePro_pars_all'] = pars_all

# Write data to disk
ipo_all.to_pickle('data/ipo_allText.pkl')
