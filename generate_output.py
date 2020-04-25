from dlib import Exploratory_Ranking
import sys

reference_path = sys.argv[1]
ranking_path = sys.argv[2]
results_path_dir = sys.argv[3]
if sys.argv[4]:
	N = int(sys.argv[4])
if sys.argv[5]:
	arvn_n = int(sys.argv[5])

model_dir = "data/best_weights"
data_dir = "data"

bRank = Exploratory_Ranking(model_dir,data_dir)

bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
bRank.write_results(results_path_dir)