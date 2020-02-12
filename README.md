# Drivers-Article-Matching
Drivers to Article and Article to Driver Matching for the SPA project

## Description
Information retrieval using neural models. 
We leverage the semantic meaning learned by neural models to match a query with an article.

In this project we are performing two main tasks. 
- Given a query, rank a list of articles
- Given an article, rank a list of queries  

based on a similarity score.

## Link to Model Weights
https://rpi.box.com/s/ym36qaofw2rk93h3t7zbpnphu25p0q7z


## Installation
Use conda to install the libraries.
```
conda install --file requirements.txt
```

The `word_tokenize` function in nltk requires punkt tokenizer. 
After installing nltk, run the following.
```bash
python -c "import nltk; nltk.download('punkt')"
```

Create the `data` folder to keep the weights.
```bash
mkdir data
```
Download the files from the link to the `data` folder.
Extract the zip file.
```bash
cd data
unzip model_weights.zip
cd ..
```

## Usage
Example use is shown in `match_fn.py`  

This file has an example article and a list of drivers.
For ranking them, run
```
python match_fn.py
```

## Using the functions
To call the ranking functions from a different script in the same folder.
```python
from match_fn import Ranker
from nltk import word_tokenize
# Directory containing checkpoint file
model_dir = "data/best_weights"
# Directory containing vocab file
data_dir = "data"
# Create the ranker object and load model weights
ranker = Ranker(model_dir, data_dir)
```

A query or an article is a list of tokenized lowercase words. Use `word_tokenize` function from nltk to tokenize a piece of text (query or article)

For query->articles, create a list of articles and rank them.
```python
# query text is a string
query = word_tokenize(query_text.lower())
# articles is a list of tokenized articles and article_text_list is a list of article texts
articles = [word_tokenize(article_text.lower()) for article_text in article_text_list]
ids = ranker.query2articles(query, articles, 5)
```

For article->queries, create a list of queries and rank them.
```python
# article text is a string
article = word_tokenize(article_text.lower())
# queries is a list of tokenized queries and query_text_list is a list of query texts
queries = [word_tokenize(query_text.lower()) for query_text in query_text_list]
# queries is a list of queries
ids = ranker.article2queries(article, queries, 5)
```
