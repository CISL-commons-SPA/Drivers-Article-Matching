from dlib import Exploratory_Ranking
import sys
reference_path = "data/covid_19_score_0.txt"
ranking_path = "data/total_filtered.jsonl"
results_path_dir = "res"
N = 1
arvn_n = 1
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)

reference_path = "data/covid_19_score_0.txt"
ranking_path = "signalmedia-1m.jsonl"
results_path_dir = "res"
N = 100
arvn_n = 100
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)

reference_path = "data/ACME_UNIQUE.txt"
ranking_path = "data/total_filtered.jsonl"
results_path_dir = "res"
N = 10
arvn_n = 10
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)
# python generate_output.py data/covid_19_score_0.txt data/total_filtered.jsonl res 1 1