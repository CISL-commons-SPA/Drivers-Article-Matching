
from dlib import Ranker
from nltk import word_tokenize
import requests
from flask import Flask, request, jsonify 
## Example use
# Directory containing checkpoint file
model_dir = "data/best_weights"
# Directory containing vocab file
data_dir = "data"

article = "Taiwan's vice president-elect William Lai will go to this week's high-profile National Prayer Breakfast in Washington, he said on Monday, an event traditionally attended by U.S. presidents and which President Donald Trump was at last year. Lai, who assumes office in May, has angered China by saying he is a realistic worker for Taiwan independence, a red line for Beijing which considers the island merely a Chinese province with no right to state-to-state relations."
drivers = ["Govt and ANSF Strategic Communication and IO Increasing",
          "Govt and Contractor Corruption and Tribal Favoritism Decreasing",
          "Govt Funding Adequacy Decreasing",
          "US Govt Support for Operation Increasing",
          "US Govt Support for Operation Decreasing",
          "Govt Funding Adequacy Increasing",
          "Fear of Govt ANSF and Coalition Repercussions Increasing"]
ids = {"article" : article, "queries" : drivers, "num" : 5}
result = requests.post('http://127.0.0.1:5000/article2queries', json = ids)   
print(result.text)


driver = 'US Govt Support for Operation Increasing'
articles = ['This article proposes appropriating $19,605,537 to the operating budget. This amount does not include appropriations by special warrant articles, which are voted on separately. If the 2020 budget does not pass, the operating budget will remain at the 2019 level of $19,323,051. (Estimated tax impact is $18 per $100,000 assessed property value). Recommended by the Select Board by a vote of 5-0','he 1918 influenza pandemic was the deadliest event in human history (50 million or more deaths, equivalent in proportion to 200 million in today’s global population). For more than a century, it has stood as a benchmark against which all other pandemics and disease emergences have been measured. We should remember the 1918 pandemic as we deal with yet another infectious-disease emergency: the growing epidemic of novel coronavirus infectious disease (Covid-19), which is caused by the severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). This virus has been spreading throughout China for at least 2 months, has been exported to at least 36 other countries, and has been seeding more than two secondary cases for every primary case. The World Health Organization has declared the epidemic a Public Health Emergency of International Concern. If public health efforts cannot control viral spread, we will soon be witnessing the birth of a fatal global pandemic.','A system created by MIT researchers could be used to automatically update factual inconsistencies in Wikipedia articles, reducing time and effort spent by human editors who now do the task manually.',"Australia is investigating more than 50 alleged war crimes by the country's special forces in Afghanistan, including the killing of civilians and prisoners, a military watchdog said on Tuesday","resident Trump on Friday issued his alternative pay plan for 2020, endorsing a 2.6% across the board pay increase for civilian federal employees, effectively ending the administration’s push for a pay freeze next year."]

ids = {"query" : driver, "articles" : articles, "num" : 5}
result = requests.post('http://127.0.0.1:5000/query2articles', json = ids)   
print(result.text)
