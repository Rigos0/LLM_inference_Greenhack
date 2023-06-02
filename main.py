from flask import Flask, render_template, request
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        model_output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=5,
            do_sample=True,
            temperature=0.7
        )
        text = tokenizer.decode(model_output[0], skip_special_tokens=True)
        return render_template('home.html', generated_text=text)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
