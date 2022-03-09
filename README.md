# RiskyData-LDA

Copyright (c) 2022 Maximilian W. Hofer & Kenneth A. Younge.

Repo maintainer: Maximilian W. Hofer ([maximilian.hofer@epfl.ch](mailto:maximilian.hofer@epfl.ch))

   AUTHOR:  Maximilian W. Hofer  
   SOURCE:  https://github.com/mxhofer/OrgSim-RL  
   LICENSE: Access to this code is provided under an MIT License.  

The RiskyData-RL platform quantifies risk disclosures in IPO prospectuses using an LDA topic model.

# Usage
## Fork repo

Create your own copy of the repository to experiment with OrgSim-RL freely.

## Install dependencies

`git clone https://github.com/mxhofer/RiskyData-LDA.git`

`pip install -r requirements.txt`

## Prepare input data

We provide the Risky Data paper data in the `data/ipo.xlsx` file. You can replace this file with your own as long as the column with text-based risk is split into paragraphs using the `---new_paragraph---` divider.

## Run pipeline

To quantify textual risk factors, run the following scripts (in order):

1. `00-preprocess.py`: cleans text data (tokenization, stemming, etc.)
   1. Input: `data/ipo.xlsx`
   2. Output: `data/ipo_allText.pkl`
2. `01-fit.py`: fits an LDA topic model and writes paragraph-level topic loadings to disk
   1. Input: `data/ipo_allText.pkl`
   2. Output: `data/ipo_allTopics.pkl`
3. `02-normalize.py`: normalizes topic loadings to firm-level risks using year and industry groups
   1. Input: `data/ipo_allTopics.pkl`
   2. Output: `data/ipo_risk.xlsx`

A note on performance: pre-processing text data and fitting the LDA topic model take time. On a MacBook Pro 2020, the pipeline takes ~2 hours to complete.

## Inspect results

The results in `data/ipo_risk.xlsx` contain the aggregate risk disclosure and the individual risk factors for each IPO firm.
