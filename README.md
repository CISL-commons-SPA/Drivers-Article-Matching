# Drivers-Article-Matching
Drivers to Article and Article to Driver Matching for the SPA project

## Description


## Link to Model Weights
https://rpi.box.com/s/ym36qaofw2rk93h3t7zbpnphu25p0q7z


## Installation

## Usage
Example use is shown in `match_fn.py`  

```python
# Create the ranker object and load model weights
ranker = Ranker(model_dir, data_dir)
```

A query or an article is a list of tokenized lowercase words. Use `word_tokenize` function from nltk to tokenize a piece of text (query or article)
```python
query = word_tokenize(query_text.lower())
article = word_tokenize(article_text.lower())
```

For query->articles
```python
# articles is a list of articles
ids = ranker.query2articles(query, articles, 5)
```

For article->queries
```python
# queries is a list of queries
ids = ranker.article2queries(article, queries, 5)
```
