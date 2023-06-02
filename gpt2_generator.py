from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_reply(prompt):
    # Prepend the instruction to the prompt.
    full_prompt = prompt + "Here is my expert answer: "

    # Encode the prompt
    input_ids = tokenizer.encode(full_prompt, return_tensors='pt')

    # Generate a response
    output = model.generate(
        input_ids,
        max_length=30,   # Adjust as needed
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        do_sample=False,
        temperature=0.4
    )

    # Decode the output
    reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return reply

# Test the function
