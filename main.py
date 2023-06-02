from gpt2_generator import *
from sentence_encoder import *

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    prompt = data.get('prompt')
    n = data.get('n')
    responses = database.find_nearest_prompts(prompt, n)
    return jsonify(responses)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt')
    response = generate_reply(prompt)
    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
