from dlib import Exploratory_Ranking
import sys
#reference_path is the path to a text file with the drivers
reference_path = sys.argv[1]
#ranking_path is the path to a .jsonl file of articles
ranking_path = sys.argv[2]
#results_path_dir is a directory where the output will go
results_path_dir = sys.argv[3]
N = int(sys.argv[4])
arvn_n = int(sys.argv[5])

model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)

bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)