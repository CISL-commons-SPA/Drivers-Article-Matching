from dlib import Exploratory_Ranking
import sys
reference_path = "data/drivers_small.txt"
ranking_path = "data/articles_small.jsonl"
results_path_dir = "res"
N = 1
arvn_n = 1
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)

reference_path = "data/drivers_large.txt"
ranking_path = "data/total_filtered.jsonl"
results_path_dir = "res"
N = 100
arvn_n = 100
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)

reference_path = "data/drivers_large.txt"
ranking_path = "data/articles_small.jsonl"
results_path_dir = "res"
N = 10
arvn_n = 10
model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)
bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)