from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

class Database:
    def __init__(self):
        self.prompts = []
        self.embeddings = []
        self.outputs = []

    def find_nearest_prompts(self, prompt: str, n: int):
        """Finds n nearest prompts from the database"""
        encoded_prompt = model.encode([prompt])
        similarity_scores = np.inner(encoded_prompt, self.embeddings)[0]

        # find the n most similar prompts based on cosine similarity
        most_similar_indices = np.argsort(similarity_scores)[::-1][:n]
        most_similar_prompts = [self.prompts[index] for index in most_similar_indices]
        return most_similar_prompts


data = Database()
sentence_list = ['I love this.', 'I hate this.', 'This is great!', "That's a cat.", "How many days will we spend here",
                 "I love all animals with four legs."]
sentence_embeddings = model.encode(sentence_list)

data.prompts = sentence_list
data.embeddings = sentence_embeddings

print(data.find_nearest_prompts("That's dog", 3))



