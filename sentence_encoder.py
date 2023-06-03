from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


class Database:
    def __init__(self):
        self.prompts = []
        self.embeddings = []
        self.outputs = []

    def find_nearest_prompts(self, prompt: str, n: int, threshold: float = 0.19):
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

    def output_from_prompt(self, prompt):
        if prompt not in self.prompts:
            return "Prompt not found"
        index = self.prompts.index(prompt)

        return self.outputs[index]


database = Database()
sentence_list = [

    'What is the power consumption of ChatGPT?',
    'Do solar powerplants produce more energy than water plants?',
    "How many days will we spend here?",
    "I love all animals with four legs.",
    "How much does Jeff Bezos lift?",
    "How powerful is this hand dryer?",
    "How can we increase the efficiency of our project?",
    "How does gravity work?",
    "How are tornadoes formed?",
    "How do they make holograms?",
    "How does a vaccine work?",
    "How can we stop climate change?",
    "How does a computer process information?",
    "How does the stock market operate?",
    "How is the internet connected worldwide?",
    "How can I improve my cooking skills?",
    "How do self-driving cars navigate?",

    "I need some advice on buying a new car.",
    "I'm so excited for the weekend.",
    "Do you believe in aliens?",
    "I'm trying to learn a new language.",
    "Can you help me with this math problem?",
    "I'm feeling stressed out.",
    "What's your favorite type of music?",
    "I just finished a great workout.",
    "I'm looking forward to seeing my friends.",
    "What's the best way to relax after a long day?",
    "I'm not a morning person.",
    "Can you recommend a good book to read?",
    "I need some fashion advice for an upcoming event.",
    "I'm feeling inspired today.",
    "What's the meaning of life?",
    "I'm thinking of starting a new business.",
    "Do you believe in fate?",
    "I'm so proud of my accomplishments.",
    "What's your favorite season?",
    "I'm feeling nostalgic.",
    "Can you teach me how to cook?",
    "I'm afraid of spiders.",
    "What's the best way to stay motivated?",
    "I'm trying to cut down on sugar.",
    "Do you have any tips for improving productivity?",
    "I love spending time in nature.",
    "I'm feeling adventurous today.",
    "What's your favorite holiday?",
    "I'm not a fan of public speaking.",
    "Can you recommend a good TV show to watch?",
    "I'm feeling lucky.",
    "What's the secret to a happy life?",
    "I love exploring new cultures.",
    "I'm trying to eat healthier.",
    "Do you have any travel recommendations?",
    "I'm feeling overwhelmed.",
    "What's the best way to overcome fear?",
    "What's your favorite movie genre?",
    "I'm learning to play the guitar.",
    "Can you recommend a good workout routine?"]


output_list = [
    'In January, ChatGPT utilized over 23 GWh of electricity, an amount substantial enough to charge 230 000 Tesla cars.',    'Solar power plants can produce a vast amount of energy, in ideal conditions around 1,000 watts per square meter. However, water plants, or hydroelectric power, are quite consistent and can produce large amounts of power, too. It depends on the specific plants in question.',
    'The duration of our stay will be subject to the itinerary planned. Based on standard durations, it could be anything between 7 to 14 days.',
    'Animals with four legs are indeed fascinating creatures, each having unique characteristics that contribute to the biodiversity of our planet.',
    "Jeff Bezos reportedly follows a rigorous workout regimen and might be able to bench press around 200 pounds, although exact numbers aren't publicly available.",
    'The power of a hand dryer can vary based on its model and brand. Typically, commercial hand dryers have an air speed of around 19,000 to 23,000 LFM (Linear Feet per Minute).',
    'To increase the efficiency of a project, streamline communication, implement project management tools, and regularly evaluate and adjust goals. These measures can potentially increase project efficiency by up to 30%.',
    'Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another. It is responsible for the structure of the universe and it operates at an acceleration of 9.8 m/s² on Earth.',
    'Tornadoes form from severe thunderstorms. They are created when warm, moist air collides with cold, dry air, causing the warm air to rise rapidly, leading to the formation of a vortex.',
    'Holograms are created by recording a light field and then projecting it in a way that reproduces the 3D presence of the original scene. This involves complex optical setups and precise control of light.',
    'A vaccine works by mimicking disease agents to stimulate the immune system, building up a defense mechanism that recognizes and fights off the real pathogen if the body encounters it later.',
    'Stopping climate change involves multiple strategies like reducing greenhouse gas emissions, promoting renewable energy, and improving energy efficiency. By implementing these measures, we can potentially reduce global warming by 2 degrees Celsius by 2100.',
    'A computer processes information by taking inputs, storing, processing, and then outputting the results. This is accomplished through a combination of hardware and software mechanisms.',
    'The stock market operates through a network of exchanges. Companies list shares of their stock on an exchange, and investors buy those shares. This process is facilitated by stock brokers.',
    'The internet is connected worldwide through a vast network of optical fiber cables, satellites, and wireless connections. These connections span continents and oceans, linking computers and servers across the globe.',
    'Improving cooking skills can be achieved by practicing various techniques, understanding flavor combinations, and experimenting with diverse ingredients. This can potentially enhance the quality of dishes by 50%.',
    'Self-driving cars navigate using a combination of sensors, cameras, and software. These tools allow the car to perceive its environment, interpret the data, and make decisions about the route to take.',
    "Sure, are you looking for something fuel efficient or with a lot of space?",
    "That's great! Do you have any special plans?",
    "It's hard to say for sure, but the universe is so vast, it seems possible!",
    "That's a fantastic goal. What language are you considering?",
    "Absolutely, what's the problem you're trying to solve?",
    "I'm sorry to hear that. Taking deep breaths and going for a walk might help.",
    "As an AI, I don't listen to music, but I can help you find music you'll love!",
    "That's excellent! Keep up the good work.",
    "That's wonderful. Friends make life a lot more fun.",
    "Perhaps a warm bath or reading a good book could help.",
    "That's okay. Everyone has different rhythms. What helps you wake up?",
    "Sure, 'To Kill a Mockingbird' is a classic and a great read.",
    "Sure, what kind of event are you attending?",
    "That's wonderful. What are you planning to do next?",
    "As an AI, I don't have beliefs. However, many believe life's meaning is what you make of it.",
    "That's exciting! Do you have a particular type of business in mind?",
    "As an AI, I don't have personal beliefs. What's your perspective?",
    "That's wonderful. It's important to acknowledge our own achievements.",
    "As an AI, I don't have a favorite season. But I can provide information on all of them!",
    "Nostalgia can be bittersweet. Is there something specific you're reminiscing about?",
    "Of course, what would you like to learn to cook?",
    "I understand. Many humans have phobias. Do you want to know more about spiders?",
    "Setting achievable goals and rewarding yourself when you reach them could be helpful.",
    "That's a healthy choice! Have you tried substituting fruit when you crave something sweet?",
    "Creating a to-do list and prioritizing tasks can be a good start.",
    "Nature can be very calming. Do you have a favorite place to visit?",
    "That's great! Are you planning a trip or an adventurous activity?",
    "As an AI, I don't have a favorite. But I can help you find information about any holiday.",
    "Public speaking is a common fear. Practice and preparation can help a lot.",
    "Sure, 'The Crown' is a popular series. It might be worth a watch.",
    "That's great. Good luck in whatever you're hoping for!",
    "Many people find happiness in doing things they love and spending time with people they care about.",
    "That sounds interesting. What culture are you exploring currently?",
    "A balanced diet and regular exercise are usually recommended for a healthy lifestyle.",
    "Sure, do you prefer relaxing beach vacations or city breaks?",
    "I'm sorry to hear that. Sometimes, making a list can help you prioritize what needs to be done.",
    "Facing your fears and gradually exposing yourself to what scares you can often help.",
    "As an AI, I don't watch movies. But I can help you find movies from any genre!",
    "That's exciting. Keep practicing and you'll improve over time!",
    "Regular exercise that combines cardio, strength training, and flexibility exercises could be good.",
    "I'm sorry to hear that. It may help to speak with a trusted friend or family member about it.",
    "Joining clubs or groups with similar interests could be a good start",
]



sentence_embeddings = model.encode(sentence_list)

database.prompts = sentence_list
database.outputs = output_list
database.embeddings = sentence_embeddings




