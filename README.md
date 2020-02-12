# Drivers-Article-Matching
Drivers to Article and Article to Driver Matching for the SPA project

## Description
In this project we are performing two main tasks. 
- Given a query, rank a list of articles
- Given an article, rank a list of queries  

based on a similarity score.

## Link to Model Weights
https://rpi.box.com/s/ym36qaofw2rk93h3t7zbpnphu25p0q7z


## Installation
Required libraries
- tensorflow
- numpy
- nltk

Use conda to install the libraries.
```
conda install <package_name>
```

The `word_tokenize` function in nltk requires punkt tokenizer. 
After installing nltk, run the following in python.
```
import nltk
nltk.download('punkt')
```

## Usage
Example use is shown in `match_fn.py`  

Download the model weights and vocabulary from the link.
After extracting the zip file, you get a `best_weights` folder.
This is the `model_dir`.
Put `vocab100` file in a folder.
The folder containing `vocab100` file is the `data_dir`.

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
