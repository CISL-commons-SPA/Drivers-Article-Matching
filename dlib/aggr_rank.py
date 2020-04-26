from dlib import Ranker
import jsonlines
import numpy as np

class Exploratory_Ranking(Ranker):
	def __init__(self, model_dir, data_dir):
		super(Exploratory_Ranking, self).__init__(model_dir, data_dir)
		self.reference_text = []
		self.ranking_text = []
		self.articles = []
		self.ARVN_vector = []
		self.ARVN_ranks = []

	def read_lines_from_text_file(self,path):
		with open(path,'r') as filer:
			lines = filer.readlines()
		lines = [line.strip().replace('\n','').replace('\r','') for line in lines]
		return lines

	def write_results(self, path):
		path = path.strip('/')
		arpath = path + '/ranked_articles_ARVN.txt'
		arvecpath = path + '/sorted_arvn_vector.txt'

		with open(arpath,'w') as filw:
			filw.writelines([ar.strip().replace('\n','').replace('\r','') + '\n' for ar in self.articles])
		with jsonlines.open(arvecpath,'w') as fw:
			fw.write([float(ar) for ar in self.ARVN_vector])


	def do_aggro_ranking(self,reference_path = '',ranking_path = '',N=10,num=10):
		self.reference_text = self.read_lines_from_text_file(reference_path)
		self.ranking_text = self.read_lines_from_text_file(ranking_path)  	

		self.articles, self.ARVN_ranks, self.ARVN_vector = self.ARVN(self.reference_text,self.ranking_text,N,num)

def main():
	
	import sys
	reference_path = sys.argv[1]
	ranking_path = sys.argv[2]
	results_path_dir = sys.argv[3]
	N = int(sys.argv[4])
	arvn_n = int(sys.argv[5])


	model_dir = "../data/best_weights"
	data_dir = "../data"

	bRank = Exploratory_Ranking(model_dir,data_dir)

	bRank.do_aggro_ranking(reference_path,ranking_path,N,arvn_n)
	bRank.write_results(results_path_dir)


if __name__ == '__main__':
	main()