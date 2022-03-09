
# Import packages
import pandas as pd
import numpy as np

# Constants
K = 20

# 2-year mapping
year_map = {
    1996: '1996-1997',
    1997: '1996-1997',
    1998: '1998-1999',
    1999: '1998-1999',
    2000: '2000-2001',
    2001: '2000-2001',
    2002: '2002-2003',
    2003: '2002-2003',
    2004: '2004-2005',
    2005: '2004-2005',
    2006: '2006-2007',
    2007: '2006-2007',
    2008: '2008-2009',
    2009: '2008-2009',
    2010: '2010-2011',
    2011: '2010-2011',
    2012: '2012-2013',
    2013: '2012-2013',
    2014: '2014-2015',
    2015: '2014-2015',
    2016: '2016-2018', # 3 years!
    2017: '2016-2018',
    2018: '2016-2018',
}

# Load data
par_all = pd.read_pickle('data/ipo_allTopics.pkl')
print('Paragraph data shape: {}'.format(par_all.shape))

ipo_data_raw = pd.read_csv('data/ipo.csv')
print('IPO data shape: {}'.format(ipo_data_raw.shape))

ipo_data = ipo_data_raw[['Issue Year', 'industryFF12']].copy()
ipo_data.columns = ['year', 'industry']

ipo_data.reset_index(inplace=True, drop=True)
print('IPO data shape after reset: {}'.format(ipo_data.shape))

# Prepare dataset

ipo_par = par_all.merge(ipo_data, how='left', left_on='id', right_index = True)
print('IPO data shape after merge: {}'.format(ipo_par.shape))

# Add 2-year groups
ipo_par['year_group'] = ipo_par['year']
ipo_par = ipo_par.replace({'year_group': year_map})
print('IPO data shape after adding year groups: {}'.format(ipo_par.shape))

# Normalize risk

# Group by year + FF industry group
FFyear_groups = ipo_par.drop(['id', 'year'], axis=1).groupby(by=['year_group', 'industry']).count()
FFyear_groups = FFyear_groups[[0]]
FFyear_groups = FFyear_groups.reset_index()
FFyear_groups = FFyear_groups.sort_values(by=0, ascending=True)

# Identify dominant topics per paragraph
ipo_data['year_group'] = ipo_data['year']
ipo_data = ipo_data.replace({'year_group': year_map})
ipo_data['id'] = ipo_data.index

# Find row-wise maxima, i.e. dominant topics per paragraph
# Why? because I make inference on paragraph-level (not document-level)
dominant_pars = np.argmax(ipo_par.iloc[:, 1: K+1].values, axis=1)
df = pd.DataFrame({'dom': dominant_pars})
df = pd.get_dummies(df['dom'])
LDA_PAR_dom = pd.concat([ipo_par[['id']], df], axis=1)

# Aggregate to document-level
df = LDA_PAR_dom.groupby(by='id').agg(sum).reset_index()
print('Without industry and year group : {}'.format(df.shape))

LDA_PAR_dom = df.merge(ipo_data[['id', 'industry', 'year_group']], how='left', left_on='id', right_on='id')
print('With industry and year group: {}'.format(LDA_PAR_dom.shape))

# FF + 2-year counts
LDA_PAR_counts = LDA_PAR_dom.drop(['id'], axis=1).groupby(by=['year_group', 'industry']).agg(sum).reset_index()

# Mean paragraph counts per document
LDA_PAR_means = LDA_PAR_dom.drop(['id'], axis=1).groupby(by=['year_group', 'industry']).agg(np.mean).reset_index()
print('Mean of groups: {}'.format(LDA_PAR_means.shape))

# STD of FF + 2-years
LDA_PAR_stds = LDA_PAR_dom.drop(['id'], axis=1).groupby(by=['industry', 'year_group']).agg(np.std).reset_index()

# Replace nan stds (from no observation in the group by) with 0
LDA_PAR_stds = LDA_PAR_stds.replace(np.nan, 0)

# Plus 1 to avoid division by zero error
cols = [i for i in range(K)]
LDA_PAR_stds[cols] = LDA_PAR_stds[cols] + 1
print('STD of groups: {}'.format(LDA_PAR_stds.shape))

# Normalize
cols_x = [str(i)+'_x' for i in range(K)]
cols_y = [str(i)+'_y' for i in range(K)]

# De-mean
LDA_PAR_demeaned = LDA_PAR_dom.merge(LDA_PAR_means, how='left', left_on=['year_group', 'industry'], right_on=['year_group', 'industry'])
df = pd.DataFrame(LDA_PAR_demeaned[cols_x].values - LDA_PAR_demeaned[cols_y].values)
df['id'] = LDA_PAR_dom['id']
df['industry'] = LDA_PAR_dom['industry']
df['year_group'] = LDA_PAR_dom['year_group']
print('After de-meaning: {}'.format(df.shape))

# Rescale
LDA_PAR_std = df.merge(LDA_PAR_stds, how='left', left_on=['industry', 'year_group'], right_on=['industry', 'year_group'])

LDA_PAR_z = pd.DataFrame(LDA_PAR_std[cols_x].values / LDA_PAR_std[cols_y].values)
LDA_PAR_z.columns = ['rf{}'.format(i) for i in range(K)]  # rename normalized risk topic loadings
LDA_PAR_z['aggregate_risk'] = LDA_PAR_z[['rf{}'.format(i) for i in range(K)]].sum(axis=1)
LDA_PAR_z['id'] = LDA_PAR_dom['id']
print('After rescaling: {}'.format(LDA_PAR_z.shape))

LDA_PAR_out = LDA_PAR_z.merge(ipo_data_raw.drop(columns=['RF_clean_paragraphs'], axis=1), how='left', left_on='id', right_index=True, )
print('Final shape: {}'.format(LDA_PAR_z.shape))

# Write to disk
LDA_PAR_out.to_excel('data/ipo_risk.xlsx')


