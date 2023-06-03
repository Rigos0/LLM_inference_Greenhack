from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


class Database:
    def __init__(self):
        self.prompts = []
        self.embeddings = []
        self.outputs = []

    def find_nearest_prompts(self, prompt: str, n: int, threshold: float = 0.23):
        """Finds n nearest prompts from the database"""
        encoded_prompt = model.encode([prompt])
        similarity_scores = np.inner(encoded_prompt, self.embeddings)[0]

        # find the n most similar prompts based on cosine similarity
        most_similar_indices = np.argsort(similarity_scores)[::-1]
        most_similar_prompts = []
        for index in most_similar_indices:
            if similarity_scores[index] >= threshold and len(most_similar_prompts) < n:
                most_similar_prompts.append(self.prompts[index])
            if len(most_similar_prompts) == n:
                break

        return most_similar_prompts


database = Database()
sentence_list = [
    'What is the power consumption of ChatGPT.',
    'Do solar powerplants produce more energy than water plants?',
    "How many days will we spend here?",
    "I love all animals with four legs.",
    "This reminds me of my childhood.",
    "What's your favorite color?",
    "Can you recommend a good restaurant nearby?",
    "I'm feeling tired today.",
    "What's the weather like?",
    "I need a vacation.",
    "I'm craving pizza right now.",
    "Tell me a joke.",
    "I can't believe it's already June!",
    "Have you read any good books lately?",
    "I wish I could travel the world.",
    "What's your favorite hobby?"]

sentence_embeddings = model.encode(sentence_list)

database.prompts = sentence_list
database.embeddings = sentence_embeddings




