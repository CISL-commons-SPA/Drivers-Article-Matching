import requests
from flask import Flask, request, jsonify	
from nltk import word_tokenize
from dlib import Ranker
import pdb

def create_app():
	app = Flask(__name__)
	model_dir = "data/best_weights"
	data_dir = "data"
	ranker = Ranker(model_dir, data_dir)

	@app.route('/query2articles', methods=['POST'])
	def q2a():
		content = request.get_json()
		query = content.get('query')
		articles = content.get("articles")
		num = content.get('num')
		query = word_tokenize(query.lower())
		articles = [word_tokenize(article.lower()) for article in articles]
		data = ranker.query2articles(query, articles, num)
		return jsonify(data)

	@app.route('/article2queries', methods=['POST'])	
	def a2q():
		content = request.get_json()
		article = content.get("article")
		queries = content.get('queries')
		num = content.get('num')
		article = word_tokenize(article.lower())
		queries = [word_tokenize(query.lower()) for query in queries]
		data = ranker.article2queries(article, queries, num)
		return jsonify(data)

	return app
