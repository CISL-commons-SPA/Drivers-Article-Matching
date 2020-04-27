# Drivers-Article-Matching
This repository contains Drivers to Article and Article to Driver Matching for the SPA project

## Description
This project is focuses on doing information retrieval by using neural models. 
We leverage the semantic meaning learned by neural models to match a query with an article and send to a POST endpoint via Flask.

In this project we are performing two main tasks, both of which are based on a similarity score: 
- Given a query, rank a list of articles
- Given an article, rank a list of queries  

This specific branch is written as per instructions from Hui as per this email:

- Driver-Article Matching(DAM): Anik's algorithm
- Driver-Article Relevance Value( DARV): Similarity output value by DAM for each Article against each Driver
- Aggregated Relevance Value (ARV): the sum of the relevance value of all items divided by the number of items
- Aggregated Relevance Value of Top N items in the ranking list(ARVN): the sum of the relevance value of top N items in the list divided by N
- Task No. 1
- Applying DAM to all 1800 against Driver(i) to get a Rank List for Driver i, which is RLD(i)
- Calculate ARVN (start with N=10) with RLD(i) and get ARVN (i)
- Rank all ARVN(i) and pick top 10 with bigger ARVN and select 10 Drivers accordingly
- Expected results
- 10 selected drivers and their corresponding top 10 articles

## Installation
First use conda to install the libraries.
```bash
conda install --file requirements.txt
```

The `word_tokenize` function in nltk requires punkt tokenizer, so after installing the libraries, run the following on the command line:
```bash
python -c "import nltk; nltk.download('punkt')"
```

The model weights can be dowloaded from the link below, and should be unzipped in the data directory:

Link to model weights: https://rpi.box.com/s/ym36qaofw2rk93h3t7zbpnphu25p0q7z

```bash
cd data
unzip model_weights.zip
cd ..
```

## Usage
To initialize the Flask server, the makefile can be run via the following command:
```bash
make run
```

To test the server after initialization, traverse to the root of the repository and execute:

```bash
make test
```

Some example use is shown in `match_fn.py`, which has has an example article and a list of drivers.
To rank them, you can run:
```bash
python dlib/match_fn.py
```

To run the Exploratory Ranking functions, you can do the following:
```python
python3 e_rank.py driver_file article_file result_directory N arvn_n
```
Where:
driver_file is the location of a text file of divers
article_file is the location of a .jsonl file with articles
result_directory is an existing directory where you want the results to go
N is the number of text elements wanted from the score vector in ARVN (default 10) 
arvn_n is the number of text elements wanted from ARVN (default 10)

## Using the functions
To call the ranking functions from a different script in the same folder, you can do the following:
```python
from dlib import Ranker
from nltk import word_tokenize
# Directory containing checkpoint file
model_dir = "data/best_weights"
# Directory containing vocab file
data_dir = "data"
# Create the ranker object and load model weights
ranker = Ranker(model_dir, data_dir)
```

A query or an article is a list of tokenized lowercase words, so the `word_tokenize` function from nltk can be used to tokenize either a query or an article, both of which are demonstrated below.

For query->articles, create a list of articles and rank them in the following way:
```python
# query text is a string
query = word_tokenize(query_text.lower())
# articles is a list of tokenized articles and article_text_list is a list of article texts
articles = [word_tokenize(article_text.lower()) for article_text in article_text_list]
data = ranker.query2articles(query, articles, 5)
```

For article->queries, create a list of queries and rank them in the following way:
```python
# article text is a string
article = word_tokenize(article_text.lower())
# queries is a list of tokenized queries and query_text_list is a list of query texts
queries = [word_tokenize(query_text.lower()) for query_text in query_text_list]
# queries is a list of queries
data = ranker.article2queries(article, queries, 5)
```