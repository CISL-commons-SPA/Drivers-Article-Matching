from nltk import word_tokenize
import numpy as np

from load_embed import Embedding

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
    Input text is list of tokens

    Args
      query: list of words
      articles: list of list of words
    Returns
      Ranked indexes to the input list of articles
    """
    qemb = self.embed.get_query_embed(query, norm=True)
    aemb = np.vstack([self.embed.get_article_embed(a[:1000]) for a in articles])
    aemb /= np.linalg.norm(aemb, axis=-1, keepdims=True)
    ids, dots = rank(qemb, aemb, num=num)
    for r in range(len(ids)):
      print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(articles[ids[r]][:50]), "\n")
    
    return ids
  
  def article2queries(self, article, queries, num=5):
    """
    Args
      queries: list of str
      article: str
    Returns
      Ranked indexes to the input list of queries
    """
    aemb = self.embed.get_article_embed(article[:1000])
    qemb = np.vstack([self.embed.get_query_embed(q) for q in queries])
    qemb /= np.linalg.norm(qemb, axis=-1, keepdims=True)
    ids, dots = rank(aemb, qemb, num=num)
    for r in range(len(ids)):
      print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(queries[ids[r]]), "\n")
    
    return ids
  
if __name__ == "__main__":
  # Example use
  # Directory containing checkpoint file
  model_dir = "experiments/signal-news/embed-rand-d768-b1000-e30-margin/best_weights"
  # Directory containing vocab file
  data_dir = "data/signal-news"
  ranker = Ranker(model_dir, data_dir)

  q = "rapid technological change".split()
  # arts = load_by_line(data_dir + "/test/articles.txt", 100)
  # article = word_tokenize("Unless more incriminating evidence emerges to dramatically alter public perception, the impeachment trial of Donald Trump is effectively over. It’s comforting, no doubt, to believe that Trump has survived this entire debacle because he possesses a tighter hold on his party than Barack Obama or George W. Bush or any other contemporary president did. But while partisanship might be corrosive, it’s also the norm. In truth, Trump, often because of his own actions, has likely engendered less loyalty than the average president, not more.".lower()
  # )
  article = word_tokenize("Taiwan's vice president-elect William Lai will go to this week's high-profile National Prayer Breakfast in Washington, he said on Monday, an event traditionally attended by U.S. presidents and which President Donald Trump was at last year. Lai, who assumes office in May, has angered China by saying he is a \"realistic worker for Taiwan independence\", a red line for Beijing which considers the island merely a Chinese province with no right to state-to-state relations.".lower()
  )
  drivers = [s.lower().split() for s in ["Govt and ANSF Strategic Communication and IO Increasing",
            "Govt and Contractor Corruption and Tribal Favoritism Decreasing",
            "Govt Funding Adequacy Decreasing",
            "US Govt Support for Operation Increasing",
            "US Govt Support for Operation Decreasing",
            "Govt Funding Adequacy Increasing",
            "Fear of Govt ANSF and Coalition Repercussions Increasing"]]

  print(" ".join(article)[:500], '\n')
  # print(ranker.query2articles(q, arts))
  print(ranker.article2queries(article, drivers, -1))