import requests
from flask import Flask, request, jsonify

def create_app():
	app = Flask(__name__)
	@app.route('/receive/<data>', methods=['POST'])

	def get_data(data):
	  data = request.get_json()
	  print(data)
	  return jsonify(data)
	return app
