# Created by Maximilian Hofer in March 2022

# Import packages
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.matutils import corpus2csc
import pandas as pd

SEED = 17

# Load data from Step-00
ipo_all = pd.read_pickle('data/ipo_allText.pkl')
ipo_all.reset_index(inplace=True, drop=True)

par_all = list(ipo_all['RF_clean_prePro_pars_all'].values)
par_all_original = list(ipo_all['RF'].values)

print('Data shape: {}'.format(ipo_all.shape))

# Create dictionary based on the full dataset
id2word = corpora.Dictionary(ipo_all['RF_clean_prePro_pars_all'].values)

# Create corpus
par_all_corpus = [id2word.doc2bow(text) for text in par_all]

# Fit model
lda = LdaModel(corpus=par_all_corpus, num_topics=20, passes=10, id2word=id2word, random_state=SEED)

modelInference = pd.DataFrame(corpus2csc(lda[par_all_corpus]).T.toarray())

print('Data shape with topics: {}'.format(modelInference.shape))

# Concatenate full sample with topic model loadings
ipo_allTopics = pd.concat([ipo_all, modelInference], axis=1)
ipo_allTopics.drop(labels=['RF', 'RF_clean_prePro_pars_all'], axis='columns', inplace=True)

print('Data shape with topics and IPO data: {}'.format(modelInference.shape))

# Write data to disk for Step-02
ipo_allTopics.to_pickle('data/ipo_allTopics.pkl')
