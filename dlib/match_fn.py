from nltk import word_tokenize
from scipy.stats import rankdata
import jsonlines
import numpy as np
import requests
import sys
pmod = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or pmod.__name__ == '__main__':
    from load_embed import Embedding
else:

    from .load_embed import Embedding

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
    # drivers = []
    results = []
    probs = []
    for r in range(len(ids)):
      results.append(" ".join(articles[ids[r]][:50]))
      probs.append(sigmoid(dots[r]))
      # print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(articles[ids[r]][:50]), "\n")
    data = [{"index" : int(ids[i]), "prob" : probs[i]} for i in range(len(results))] 
    return data
  
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
      # print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(queries[ids[r]]), "\n")
    data = [{"driver" : drivers[i], "prob" : probs[i]} for i in range(len(drivers))] 
    return data

  def accumulate_text(self, ranker, queries, articles, num = 10):
    ranked_articles = []
    # allvalues = [[] for i in range(len(queries))]
    allvalues = np.ndarray(shape = (len(queries), len(articles)))
    allrankings = [] * len(queries)
    # sumvalues = [0] * len(articles)
    sumvalues = np.zeros(len(articles))
    for q in queries:
      data = ranker.query2articles(q, articles, len(articles))
      allrankings.append(data)
      for d in data:
        sumvalues[d['index']] += d['prob']
    sortedarticles = rankdata(sumvalues, method = 'max')
    c = 0
    for s in sortedarticles:
      ranked_articles.append(articles[s - 1])
      for i in range(len(queries)):
        for key in allrankings[i]:
          if key["index"] == s - 1:
            allvalues[i][c] = key["prob"]
            break
      c += 1
    sumvalues[::-1].sort()
    print("Here are the articles in order:", ranked_articles[:num])
    print("Here are the sums of probabilities ranked:", sumvalues[:num])
    print("Here is the matrix of data:", allvalues[:num])
    return

  
if __name__ == "__main__":
    ## Example use
    # Directory containing checkpoint file
    model_dir = "../data/best_weights"
    # Directory containing vocab file
    data_dir = "../data"
    # Create the ranker object and load model weights
    ranker = Ranker(model_dir, data_dir)

    article = word_tokenize("Taiwan's vice president-elect William Lai will go to this week's high-profile National Prayer Breakfast in Washington, he said on Monday, an event traditionally attended by U.S. presidents and which President Donald Trump was at last year. Lai, who assumes office in May, has angered China by saying he is a \"realistic worker for Taiwan independence\", a red line for Beijing which considers the island merely a Chinese province with no right to state-to-state relations.".lower()
    )
    # drivers = [s.lower().split() for s in ["Govt and ANSF Strategic Communication and IO Increasing",
    #           "Govt and Contractor Corruption and Tribal Favoritism Decreasing",
    #           "Govt Funding Adequacy Decreasing",
    #           "US Govt Support for Operation Increasing",
    #           "US Govt Support for Operation Decreasing",
    #           "Govt Funding Adequacy Increasing",
    #           "Fear of Govt ANSF and Coalition Repercussions Increasing"]]
    # drivers = [s.lower().split() for s in ["Covid-19 is spreading",
    #           "I had pizza for lunch today",
    #           "it is 40 degrees outside",
    #           "my ai homework is due next friday",
    #           "I am a junior at RPI",
    #           "I'm tired",
    #           "six plus seven equals thirteen"]]
    """
    Create obvious drivers related to COVID, driv_A
    Create positive articles which are definitely related, set_A
    For negative articles pick out first 5k articles in signal news, set_B
    Rank [set_A + set_B] with driv_A
    """
    driv_A = [s.lower().split() for s in ["Covid-19",
              "infectious disease outbreaks",
              "high costs of treatment",
              "tighter quarantine",
              "pandemic",
              "concern for future outbreaks",
              "fear of disease spread"]]
    # ids = {"article" : article, "queries" : drivers, "num" : 5}
    # articles = ['This article proposes appropriating $19,605,537 to the operating budget. This amount does not include appropriations by special warrant articles, which are voted on separately. If the 2020 budget does not pass, the operating budget will remain at the 2019 level of $19,323,051. (Estimated tax impact is $18 per $100,000 assessed property value). Recommended by the Select Board by a vote of 5-0','he 1918 influenza pandemic was the deadliest event in human history (50 million or more deaths, equivalent in proportion to 200 million in today’s global population). For more than a century, it has stood as a benchmark against which all other pandemics and disease emergences have been measured. We should remember the 1918 pandemic as we deal with yet another infectious-disease emergency: the growing epidemic of novel coronavirus infectious disease (Covid-19), which is caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This virus has been spreading throughout China for at least 2 months, has been exported to at least 36 other countries, and has been seeding more than two secondary cases for every primary case. The World Health Organization has declared the epidemic a Public Health Emergency of International Concern. If public health efforts cannot control viral spread, we will soon be witnessing the birth of a fatal global pandemic.','A system created by MIT researchers could be used to automatically update factual inconsistencies in Wikipedia articles, reducing time and effort spent by human editors who now do the task manually.',"Australia is investigating more than 50 alleged war crimes by the country's special forces in Afghanistan, including the killing of civilians and prisoners, a military watchdog said on Tuesday","resident Trump on Friday issued his alternative pay plan for 2020, endorsing a 2.6% across the board pay increase for civilian federal employees, effectively ending the administration’s push for a pay freeze next year."]
    # articles = ["Wow, I cant believe Covid-19 is spreading this fast", "test2, dont pick this one either", "the spreading of Covid-19 is unprecedented, we are preparing for the worst", "test (please dont pick this one)"]
    # set_A = ["Wow, I cant believe the infectious disease Covid-19 is spreading this fast", "The cost to treat Covid-19 is extremely expensive, and I fear that this outbreak will spread further", "the spreading of Covid-19 is unprecedented, we are preparing for the worst", "This pandemic is bad, I hope the outbreaks in the future are not this bad", "I thought that quarantine couldn't get any stricter"]
    # set_A = ['disease future Covid-19 Covid-19 concern spread infectious pandemic treatment for concern of tighter tighter spread of of for of outbreaks spread costs fear fear costs disease infectious disease high for high disease concern treatment tighter fear treatment infectious disease quarantine Covid-19 of outbreaks pandemic quarantine future future high outbreaks costs of outbreaks outbreaks outbreaks quarantine pandemic disease', 'high disease of fear concern outbreaks spread future Covid-19 tighter high treatment infectious future outbreaks of spread Covid-19 infectious tighter disease pandemic tighter outbreaks concern of of infectious future Covid-19 pandemic fear for costs fear quarantine concern of for outbreaks spread disease of outbreaks costs disease costs for treatment quarantine high disease quarantine pandemic outbreaks treatment disease', 'of spread costs future future tighter Covid-19 tighter costs disease high Covid-19 of disease disease tighter of spread disease treatment concern outbreaks infectious outbreaks high future infectious of fear of infectious Covid-19 outbreaks outbreaks quarantine outbreaks fear disease for concern pandemic treatment fear spread costs for quarantine outbreaks pandemic quarantine pandemic concern for of high disease treatment', 'treatment disease high of for disease high disease pandemic Covid-19 treatment infectious infectious quarantine future treatment pandemic costs disease quarantine Covid-19 outbreaks fear infectious tighter Covid-19 concern tighter disease of spread outbreaks tighter fear future of outbreaks of spread spread high outbreaks pandemic for outbreaks costs quarantine future of costs disease for concern concern outbreaks of fear', 'future costs quarantine for pandemic of disease treatment quarantine outbreaks spread Covid-19 infectious high concern costs concern disease outbreaks of fear spread spread tighter concern outbreaks infectious of tighter disease for Covid-19 outbreaks pandemic pandemic outbreaks quarantine disease disease fear future disease outbreaks fear costs treatment high of tighter infectious of of Covid-19 future treatment for high']
    set_A = ["Covid-19",
              "infectious disease outbreaks",
              "high costs of treatment",
              "tighter quarantine",
              "pandemic",
              "concern for future outbreaks",
              "fear of disease spread"]
    set_B = []
    i = 0
    with jsonlines.open(data_dir + "/signalmedia-1m.jsonl") as f:
      for j in f:
        if i != 5000:
          set_B.append(j["content"])
          i += 1
        else:
          break
    articles = set_A + set_B
    for a in articles:
      a = word_tokenize(a)
    ranker.accumulate_text(ranker, driv_A, articles, 50)
    # result = requests.post('http://127.0.0.1:5000/article2queries', json = ids)   
    # print(result.text)
