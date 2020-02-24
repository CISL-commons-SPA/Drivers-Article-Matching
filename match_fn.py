from nltk import word_tokenize
import numpy as np
import requests
import flask_server
from load_embed import Embedding
from flask import Flask, request, jsonify

def load_by_line(path_to_file, max_lines=-1):
  lines = []
  with open(path_to_file) as f:
    for i, line in enumerate(f):
      lines.append(line.strip().lower().split()[:1000])
      if i == max_lines - 1:
        break
  return lines

def cosine_sim(a,b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def rank(qemb, aembs, num=5):
  adist = np.dot(aembs, qemb.T)
  highsim_idxs = adist.argsort()[::-1]
  highsim_idxs = highsim_idxs[:num]
  return highsim_idxs, adist[highsim_idxs]

class Ranker():

  def __init__(self, model_dir, data_dir):
    """
    Args
      model_dir: str, directory containing the model weights
      data_dir: str, directory containing the vocabulary
    """
    self.embed = Embedding(model_dir, data_dir)

  def query2articles(self, query, articles, num=5):
    """
    Takes in a tokenized query and a list of tokenized articles.
    Return top-ranked 'num' articles from the input list.

    Args
      query: list, tokenized query
      articles: list, list of tokenized articles
      num: number of articles to return
    Returns
      Ranked indexes to the input list of articles
    """
    qemb = self.embed.get_query_embed(query, norm=True)
    aemb = np.vstack([self.embed.get_article_embed(a[:1000]) for a in articles])
    aemb /= np.linalg.norm(aemb, axis=-1, keepdims=True)
    ids, dots = rank(qemb, aemb, num=num)
    drivers = []
    probs = []
    for r in range(len(ids)):
      drivers.append(" ".join(articles[ids[r]][:50]))
      probs.append(sigmoid(dots[r]))
      print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(articles[ids[r]][:50]), "\n")
    data = [{"driver" : drivers[i], "prob" : probs[i]} for i in range(len(drivers))] 
    res = requests.post('http://127.0.0.1:5000/receive/<data>', json = data)
    return ids
  
  def article2queries(self, article, queries, num=5):
    """
    Takes in a tokenized article and a list of tokenized queries.
    Return top-ranked 'num' queries from the input list.

    Args
      queries: list, tokenized article
      article: list, list of tokenized query
      num: number of queries to return
    Returns
      Ranked indexes to the input list of queries
    """
    aemb = self.embed.get_article_embed(article[:1000])
    qemb = np.vstack([self.embed.get_query_embed(q) for q in queries])
    qemb /= np.linalg.norm(qemb, axis=-1, keepdims=True)
    ids, dots = rank(aemb, qemb, num=num)
    drivers = []
    probs = []
    for r in range(len(ids)):
      drivers.append(" ".join(queries[ids[r]]))
      probs.append(sigmoid(dots[r]))
      print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(queries[ids[r]]), "\n")
    data = [{"driver" : drivers[i], "prob" : probs[i]} for i in range(len(drivers))] 
    res = requests.post('http://127.0.0.1:5000/receive/<data>', json = data)
    return ids
  
if __name__ == "__main__":
  app = flask_server.create_app()
  with app.app_context():
    ## Example use
    # Directory containing checkpoint file
    model_dir = "data/best_weights"
    # Directory containing vocab file
    data_dir = "data"
    # Create the ranker object and load model weights
    ranker = Ranker(model_dir, data_dir)

    article = word_tokenize("Taiwan's vice president-elect William Lai will go to this week's high-profile National Prayer Breakfast in Washington, he said on Monday, an event traditionally attended by U.S. presidents and which President Donald Trump was at last year. Lai, who assumes office in May, has angered China by saying he is a \"realistic worker for Taiwan independence\", a red line for Beijing which considers the island merely a Chinese province with no right to state-to-state relations.".lower()
    )
    drivers = [s.lower().split() for s in ["Govt and ANSF Strategic Communication and IO Increasing",
              "Govt and Contractor Corruption and Tribal Favoritism Decreasing",
              "Govt Funding Adequacy Decreasing",
              "US Govt Support for Operation Increasing",
              "US Govt Support for Operation Decreasing",
              "Govt Funding Adequacy Increasing",
              "Fear of Govt ANSF and Coalition Repercussions Increasing"]]

    ids = ranker.article2queries(article, drivers, 5)