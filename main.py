from gpt3_generator import *
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

@app.route('/fetch_saved_reply', methods=['POST'])
def fetch_saved_reply():
    data = request.get_json()
    suggestion = data.get('suggestion')
    # Fetch the saved reply from the database using the suggestion as a query
    # This is just an example, you will need to implement the actual database query
    #saved_reply = database.fetch_saved_reply(suggestion)
    saved_reply = "mockup"
    return jsonify(saved_reply)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
