
from dlib import Ranker
from nltk import word_tokenize
import requests
from flask import Flask, request, jsonify 
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
ids = {"article" : article, "queries" : drivers, "num" : 5}
result = requests.post('http://127.0.0.1:5000/article2queries', json = ids)   
print(result.text)