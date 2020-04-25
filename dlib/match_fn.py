from nltk import word_tokenize
from scipy.stats import rankdata
import numpy as np
import pdb
import jsonlines
import requests
import pandas as pd
import sys
import random

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
      Ranked indices to the input list of articles
    """
    qemb = self.embed.get_query_embed(query, norm=True)
    aemb = np.vstack([self.embed.get_article_embed(a[:1000]) for a in articles])
    aemb /= np.linalg.norm(aemb, axis=-1, keepdims=True)
    ids, dots = rank(qemb, aemb, num=num)
    # drivers = []
    results = []
    probs = []
    for r in range(len(ids)):
      results.append("".join(articles[ids[r]][:50]))
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
      Ranked indices to the input list of queries
    """
    aemb = self.embed.get_article_embed(article[:1000])
    qemb = np.vstack([self.embed.get_query_embed(q) for q in queries])
    qemb /= np.linalg.norm(qemb, axis=-1, keepdims=True)
    ids, dots = rank(aemb, qemb, num=num)
    drivers = []
    probs = []
    for r in range(len(ids)):
      drivers.append("".join(queries[ids[r]]))
      probs.append(sigmoid(dots[r]))
      # print("Rank %d: Index %d, prob = %.2f" %(r+1, ids[r], sigmoid(dots[r])), " ".join(queries[ids[r]]), "\n")
    data = [{"driver" : drivers[i], "prob" : probs[i]} for i in range(len(drivers))] 
    return data

  def rank_vector(self,array,ascending=0):
    """
    Takes in an array and returns the ranked assigned as per ascending or descening order
    """
    return pd.DataFrame.rank(pd.DataFrame(array),ascending=ascending,method='first').to_numpy(int).ravel()

  def accumulate_text(self, queries, articles, num = 10):
    ranked_articles = []
    allvalues = np.ndarray(shape = (len(queries), len(articles)))
    sumvalues = np.zeros(len(articles))
    for i,q in enumerate(queries):
      data = self.query2articles(word_tokenize(q), [word_tokenize(a) for a in articles], len(articles))
      for d in data:
        # print(d)
        sumvalues[d['index']] += d['prob']
        allvalues[i][d['index']] = d['prob']
      print(i)
    sortedarticles = self.rank_vector(sumvalues) - 1
    ranked_articles = np.array(articles)[sortedarticles].tolist()
    sumvalues[::-1].sort()
    return ranked_articles[:num],sumvalues[:num],allvalues[:num],sortedarticles[:num]
  
  def ARVN(self,queries,articles,N=10,n_elements=10):
    _,_,score_matrix,_ = self.accumulate_text(queries,articles,len(articles))
    for i in range(score_matrix.shape[0]):
      score_matrix[i,:][::-1].sort()
    score_matrix = score_matrix[:,:N]
    ARVN_vector = score_matrix.mean(axis=1)
    ARVN_ranks = self.rank_vector(ARVN_vector) - 1
    rettext = np.array(queries)[ARVN_ranks][:n_elements].tolist()
    ARVN_vector[::-1].sort()
    return rettext, ARVN_ranks, ARVN_vector

  def accumulate_text_rank(self, ranker, queries, articles, num = 10):
    np.set_printoptions(threshold=sys.maxsize)
    ranked_articles = []
    allvalues = np.ndarray(shape = (len(queries), len(articles)))
    allrankings = [] * len(queries)
    sumvalues = np.zeros(len(articles))
    for q in queries:
      data = ranker.query2articles(q, articles, len(articles))
      allrankings.append(data)
      for d in data:
        sumvalues[d['index']] += d['index']
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
    sumvalues = np.sort(sumvalues)
    print("Here are the articles in order:", ranked_articles[:num])
    print("Here are the sums of indices ranked:", sumvalues[:num])
    print("Here is the matrix of data:", allvalues[:num])
    f1 = open("ranked_articles.txt", 'w')
    f2 = open("sumvalues.txt", 'w')
    f3 = open("allvalues.txt", 'w')
    for l in ranked_articles[:num]:
      f1.write('%s\n' % l)
    for l in sumvalues[:num]:
      f2.write('%s\n' % l)
    for l in allvalues[:num]:
      f3.write('%s\n' % l)

    f1.close()
    f2.close()
    f3.close()
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
    driv_A = ["coronavirus outbreak",
              "infectious disease outbreaks",
              "high costs of treatment",
              "tighter quarantine",
              "pandemic",
              "concern for future outbreaks",
              "fear of disease spread"]
    # ids = {"article" : article, "queries" : drivers, "num" : 5}
    # articles = ['This article proposes appropriating $19,605,537 to the operating budget. This amount does not include appropriations by special warrant articles, which are voted on separately. If the 2020 budget does not pass, the operating budget will remain at the 2019 level of $19,323,051. (Estimated tax impact is $18 per $100,000 assessed property value). Recommended by the Select Board by a vote of 5-0','he 1918 influenza pandemic was the deadliest event in human history (50 million or more deaths, equivalent in proportion to 200 million in today’s global population). For more than a century, it has stood as a benchmark against which all other pandemics and disease emergences have been measured. We should remember the 1918 pandemic as we deal with yet another infectious-disease emergency: the growing epidemic of novel coronavirus infectious disease (Covid-19), which is caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This virus has been spreading throughout China for at least 2 months, has been exported to at least 36 other countries, and has been seeding more than two secondary cases for every primary case. The World Health Organization has declared the epidemic a Public Health Emergency of International Concern. If public health efforts cannot control viral spread, we will soon be witnessing the birth of a fatal global pandemic.','A system created by MIT researchers could be used to automatically update factual inconsistencies in Wikipedia articles, reducing time and effort spent by human editors who now do the task manually.',"Australia is investigating more than 50 alleged war crimes by the country's special forces in Afghanistan, including the killing of civilians and prisoners, a military watchdog said on Tuesday","resident Trump on Friday issued his alternative pay plan for 2020, endorsing a 2.6% across the board pay increase for civilian federal employees, effectively ending the administration’s push for a pay freeze next year."]
    # articles = ["Wow, I cant believe Covid-19 is spreading this fast", "test2, dont pick this one either", "the spreading of Covid-19 is unprecedented, we are preparing for the worst", "test (please dont pick this one)"]
    set_A = ["Wow, I cant believe the infectious disease Covid-19 is spreading this fast", 
            "The cost to treat viral disease is extremely expensive, and I fear that this outbreak will spread further", 
            "the spreading of viral outbreak is unprecedented, we are preparing for the worst", 
            "This pandemic is bad, I hope the outbreaks in the future are not this bad", 
            "I thought that quarantine for the disease couldn't get any stricter"]
    set_B = []
    i = 0
    with jsonlines.open(data_dir + "/signalmedia-1m.jsonl") as file:
      for line in file:
        content = line['content']
        content = content.strip().replace('\n','').replace('\r','').replace(u'\xa0','')
        if(len(word_tokenize(content)) > 100):
          set_B.append(line["content"])
          i+=1
        if(i == 5000):
          break
    articles = set_A + set_B
    articles = random.sample([ar.strip().replace('\n','').replace('\r','').replace(u'\xa0','') for ar in articles if ar != ''],len(articles))
    with open('../res/original_articles.txt','w') as ori:
      ori.writelines([ar.strip().replace('\n','') + '\n' for ar in articles])
    with open('../res/original_drivers.txt','w') as dri:
      dri.writelines([ar.strip().replace('\n','') + '\n' for ar in driv_A])    
    print('obtained data!')
    ranked_articles,sumvals,allvalues,sortedarticles = ranker.accumulate_text(driv_A, articles, 30)
    # try:
    #   with open('../res/ranked_articles.txt','w') as res:
    #     res.writelines([r.strip().replace('\n','') + '\n' for r in ranked_articles])
    #   with jsonlines.open('../res/sumvalues.txt','w') as sumval:
    #     sumval.write_all(sumvals) 
    #   with jsonlines.open('../res/allvalues.txt','w') as allval:
    #     allval.write_all([val.tolist() for val in allvalues])
    #   with jsonlines.open('../res/sortedarticles.txt','w') as sortart:
    #     sortart.write([srt for srt in sortedarticles.astype('float')])
    # except:
    #   pdb.set_trace()
    articles, ARVN_ranks, ARVN_vector = ranker.ARVN(articles,driv_A,10,10)
    try:
      with open('../res/ARVN/ranked_articles.txt','w') as filew:
        filew.writelines([ar.replace('\n','').replace('\r','') + '\n' for ar in articles])
      with jsonlines.open('../res/ARVN/ARVN_vector.txt','w') as av:
        av.write([float(a) for a in ARVN_vector])
    except:
      pdb.set_trace()
    # print(ranked_articles,sumvals,allvalues,sortedarticles)