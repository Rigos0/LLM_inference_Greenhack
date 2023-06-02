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
            print(similarity_scores[index])
            if similarity_scores[index] >= threshold and len(most_similar_prompts) < n:
                most_similar_prompts.append(self.prompts[index])
            if len(most_similar_prompts) == n:
                break

        return most_similar_prompts


database = Database()
sentence_list = [
    'I love this.',
    'I hate this.',
    'This is great!',
    "That's a cat.",
    "How many days will we spend here?",
    "I love all animals with four legs.",
    "This reminds me of my childhood.",
    "What's your favorite color?",
    "Can you recommend a good restaurant nearby?",
    "I'm feeling tired today.",
    "What's the weather like?",
    "I need a vacation.",
    "What's your opinion on the latest movie?",
    "I'm craving pizza right now.",
    "Tell me a joke.",
    "I can't believe it's already June!",
    "Have you read any good books lately?",
    "I wish I could travel the world.",
    "What's your favorite hobby?",
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
    "Can you recommend a good workout routine?",
    "I'm feeling lost and don't know what to do.",
    "What's the best way to make friends as an adult?",
    "I love going to concerts.",
    "I'm feeling grateful for everything in my life.",
    "What's your favorite quote?",
    "I'm trying to be more mindful.",
    "Can you recommend a good podcast to listen to?",
    "I'm feeling stuck in a rut.",
    "What's the best way to handle conflicts?",
    "I love going for long walks in nature.",
    "I'm feeling anxious about the future.",
    "What's the best way to overcome procrastination?",
    "I'm trying to find my passion in life.",
    "Can you recommend a good documentary to watch?",
    "I'm feeling overwhelmed with work.",
    "What's your favorite type of cuisine?",
    "I love watching sunsets.",
    "I'm feeling motivated to achieve my goals.",
    "What's the best way to stay organized?",
    "I'm trying to improve my public speaking skills.",
    "Can you recommend a good self-help book?",
    "I'm feeling grateful for the little things in life.",
    "What's your favorite dessert?",
    "I love dancing like nobody's watching.",
    "I'm feeling excited about the possibilities ahead.",
    "What's the best way to handle rejection?",
    "I'm trying to be more environmentally conscious.",
    "Can you recommend a good meditation app?",
    "I'm feeling lonely and in need of companionship.",
    "What's the best way to build self-confidence?",
    "I love exploring new hiking trails.",
    "I'm feeling curious about the mysteries of the universe.",
    "What's the best way to start a morning routine?",
    "I'm trying to improve my time management skills.",
    "Can you recommend a good comedy movie?",
    "I'm feeling overwhelmed with information overload.",
    "What's the best way to overcome writer's block?",
    "I love spending time with my family and loved ones.",
    "I'm feeling reflective about the past year.",
    "What's the best way to bounce back from failure?",
    "I'm trying to live a more minimalist lifestyle.",
    "Can you recommend a good productivity app?",
    "I'm feeling determined to achieve my dreams.",
    "What's the best way to stay focused?",
    "I love exploring art galleries and museums.",
    "I'm feeling hopeful about the future.",
    "What's the best way to handle criticism?",
    "I'm trying to practice gratitude daily.",
    "Can you recommend a good comedy TV series?",
    "I'm feeling overwhelmed by clutter in my life.",
    "What's the best way to cultivate patience?",
    "I love trying new recipes in the kitchen.",
    "I'm feeling inspired by nature's beauty.",
    "What's the best way to overcome self-doubt?",
    "I'm trying to live a more balanced lifestyle.",
    "Can you recommend a good sci-fi novel?",
    "I'm feeling motivated to make positive changes.",
    "What's the best way to find work-life balance?",
    "I love going for long drives to clear my mind.",
    "I'm feeling optimistic about the future.",
    "What's the best way to deal with negative people?",
    "I'm trying to practice more self-care.",
    "Can you recommend a good fantasy book?",
    "I'm feeling determined to achieve my goals.",
    "What's the best way to handle stress?",
    "I love learning about different cultures.",
    "I'm feeling inspired by the success stories of others.",
    "What's the best way to overcome social anxiety?",
    "I'm trying to be more present in the moment.",
    "Can you recommend a good historical movie?",
    "I'm feeling motivated to improve my health.",
    "What's the best way to foster creativity?",
    "I love going on road trips to explore new places.",
    "I'm feeling content with where I am in life.",
    "What's the best way to deal with toxic relationships?",
    "I'm trying to practice more mindfulness.",
    "Can you recommend a good thriller novel?",
    "I'm feeling determined to overcome obstacles.",
    "What's the best way to handle work-related stress?",
    "I love capturing moments through photography.",
    "I'm feeling inspired by the power of music.",
    "What's the best way to boost self-esteem?",
    "I'm trying to live a more eco-friendly lifestyle.",
    "Can you recommend a good romantic comedy movie?",
    "I'm feeling motivated to chase my dreams.",
    "What's the best way to cultivate a positive mindset?",
    "I love going on adventures in nature.",
    "I'm feeling grateful for the opportunities in my life.",
    "What's the best way to deal with a difficult person?",
    "I'm trying to practice more self-reflection.",
    "Can you recommend a good motivational book?",
    "I'm feeling determined to make a difference.",
    "What's the best way to handle work-life integration?",
    "I love watching the stars on a clear night.",
    "I'm feeling inspired by the resilience of others.",
    "What's the best way to overcome negative self-talk?",
    "I'm trying to simplify my life and declutter.",
    "Can you recommend a good mystery novel?",
    "I'm feeling motivated to create positive change.",
    "What's the best way to embrace uncertainty?",
    "I love attending live music concerts.",
    "I'm feeling grateful for the lessons life has taught me.",
    "What's the best way to handle a setback?",
    "I'm trying to find balance in all aspects of my life.",
    "Can you recommend a good documentary series?",
    "I'm feeling determined to reach new heights.",
    "What's the best way to cultivate resilience?",
    "I love going on spontaneous adventures.",
    "I'm feeling inspired by the beauty of nature.",
    "What's the best way to overcome analysis paralysis?",
    "I'm trying to practice more self-compassion.",
    "Can you recommend a good action-packed movie?",
    "I'm feeling motivated to step out of my comfort zone.",
    "What's the best way to handle imposter syndrome?",
    "I love exploring new recipes and cooking techniques.",
    "I'm feeling grateful for the support of my loved ones.",
    "What's the best way to embrace change?",
    "I'm trying to find joy in the little things.",
    "Can you recommend a good science fiction TV series?",
    "I'm feeling determined to make a positive impact.",
    "What's the best way to overcome fear of failure?",
    "I love going for long hikes in the mountains.",
    "I'm feeling inspired by the creativity of others.",
    "What's the best way to develop a growth mindset?",
    "I'm trying to live a more mindful and intentional life.",
    "Can you recommend a good fantasy novel?",
    "I'm feeling motivated to pursue my passions.",
    "What's the best way to handle self-criticism?",
    "I love exploring new cities and cultures.",
    "I'm feeling grateful for the opportunities that come my way.",
    "What's the best way to navigate through uncertainty?",
    "I'm trying to practice more self-care and self-love.",
    "Can you recommend a good animated movie?",
    "I'm feeling determined to overcome challenges.",
    "What's the best way to maintain work-life harmony?",
    "I love going to art exhibitions and galleries.",
    "I'm feeling inspired by the resilience of the human spirit.",
    "What's the best way to overcome perfectionism?",
    "I'm trying to live a more sustainable lifestyle.",
    "Can you recommend a good motivational podcast?",
    "I'm feeling motivated to pursue my dreams and passions.",
    "What's the best way to overcome self-limiting beliefs?",
    "I love going on photography expeditions in nature.",
    "I'm feeling grateful for the present moment.",
    "What's the best way to handle setbacks and failures?",
    "I'm trying to find balance between work and personal life.",
    "Can you recommend a good historical fiction novel?",
    "I'm feeling determined to make a positive change in the world.",
    "What's the best way to cultivate a sense of purpose?",
    "I love exploring new hiking trails and national parks.",
    "I'm feeling inspired by the beauty of the natural world.",
    "What's the best way to overcome doubt and uncertainty?",
    "I'm trying to practice more self-reflection and self-improvement.",
    "Can you recommend a good romantic drama movie?",
    "I'm feeling motivated to take risks and embrace new opportunities.",
    "What's the best way to handle self-doubt and insecurities?",
    "I love going on road trips to discover hidden gems.",
    "I'm feeling grateful for the people who have positively influenced my life.",
    "What's the best way to navigate through challenging times?",
    "I'm trying to live a more authentic and fulfilling life.",
    "Can you recommend a good sci-fi TV series?",
    "I'm feeling determined to overcome obstacles and achieve success.",
    "What's the best way to maintain a positive mindset?",
    "I love exploring different cuisines and trying new foods.",
    "I'm feeling inspired by the resilience of the human spirit.",
    "What's the best way to overcome fear and step out of my comfort zone?",
    "I'm trying to practice more self-compassion and self-acceptance.",
    "Can you recommend a good psychological thriller novel?",
    "I'm feeling motivated to pursue my passions and make a difference.",
    "What's the best way to handle criticism and feedback?",
    "I love going on outdoor adventures and exploring nature's wonders.",
    "I'm feeling grateful for the simple pleasures in life.",
    "What's the best way to cultivate a positive work environment?",
    "I'm trying to live a more meaningful and purposeful life.",
    "Can you recommend a good crime drama movie?",
    "I'm feeling determined to overcome challenges and achieve my goals.",
    "What's the best way to maintain a healthy work-life balance?",
    "I love going on nature walks and observing wildlife.",
    "I'm feeling inspired by the power of human connection.",
    "What's the best way to overcome self-doubt and fear of judgment?",
    "I'm trying to practice more self-care and prioritize my well-being.",
    "Can you recommend a good adventure novel?",
    "I'm feeling motivated to make positive changes in my life.",
    "What's the best way to handle stress and avoid burnout?",
    "I love going on camping trips and stargazing at night.",
    "I'm feeling grateful for the opportunities that come my way.",
    "What's the best way to overcome creative blocks?",
    "I'm trying to live a more balanced and harmonious life.",
    "Can you recommend a good animated movie for all ages?",
    "I'm feeling determined to overcome obstacles and achieve success.",
    "What's the best way to cultivate resilience and bounce back from setbacks?",
    "I love exploring new hiking trails and connecting with nature.",
    "I'm feeling inspired by the beauty and wonders of the natural world.",
    "What's the best way to overcome self-criticism and cultivate self-compassion?",
    "I'm trying to practice more mindfulness and be present in the moment.",
    "Can you recommend a good science fiction TV series with complex characters?",
    "I'm feeling motivated to step out of my comfort zone and embrace new experiences.",
    "What's the best way to handle imposter syndrome and believe in my abilities?",
    "I love going on photography adventures and capturing the essence of different places.",
    "I'm feeling grateful for the supportive people in my life who lift me up.",
    "What's the best way to embrace change and adapt to new circumstances?",
    "I'm trying to practice more self-reflection and self-growth to become the best version of myself.",
    "Can you recommend a good fantasy novel with a captivating world?",
    "I'm feeling motivated to pursue my dreams with passion and perseverance.",
    "What's the best way to handle setbacks and turn them into learning opportunities?",
    "I love going on spontaneous road trips and exploring hidden gems.",
    "I'm feeling inspired by the stories of resilience and triumph against all odds.",
    "What's the best way to develop a growth mindset and embrace continuous learning?",
    "I'm trying to live a more mindful and intentional life, focusing on what truly matters.",
    "Can you recommend a good romantic comedy movie with heartwarming moments?",
    "I'm feeling determined to make positive changes in my life and leave a lasting impact.",
    "What's the best way to overcome self-limiting beliefs and unlock my full potential?",
    "I love going on hiking adventures in breathtaking natural landscapes.",
    "I'm feeling grateful for the opportunities that come my way and the lessons they bring.",
    "What's the best way to handle uncertainty and embrace the journey of the unknown?",
    "I'm trying to practice more self-care and prioritize my mental, physical, and emotional well-being.",
    "Can you recommend a good historical fiction novel that transports me to another era?",
    "I'm feeling motivated to chase my dreams with unwavering determination and perseverance.",
    "What's the best way to maintain work-life balance in a fast-paced world?",
    "I love going on outdoor expeditions and immersing myself in the beauty of nature.",
    "I'm feeling inspired by the resilience of the human spirit and the capacity for growth.",
    "What's the best way to overcome fear and step outside my comfort zone?",
    "I'm trying to practice more self-compassion and cultivate a positive self-image.",
    "Can you recommend a good thriller movie that keeps me on the edge of my seat?",
    "I'm feeling motivated to make a positive impact in my community and create meaningful change.",
    "What's the best way to handle self-doubt and build unwavering self-confidence?",
    "I love exploring diverse cultures and experiencing the richness of different traditions.",
    "I'm feeling grateful for the simple moments that bring joy and meaning to my life.",
    "What's the best way to navigate through life's challenges with grace and resilience?",
    "I'm trying to live a more mindful and conscious life, making intentional choices that align with my values.",
    "Can you recommend a good science fiction novel that explores thought-provoking concepts?",
    "I'm feeling motivated to pursue my passions with dedication and unwavering commitment.",
    "What's the best way to handle criticism and turn it into an opportunity for growth?",
    "I love going on solo adventures and embracing the freedom of exploring new places.",
    "I'm feeling inspired by the stories of those who have overcome adversity and achieved greatness.",
    "What's the best way to overcome self-doubt and believe in my own abilities?",
    "I'm trying to practice more self-care and prioritize my well-being in a hectic world.",
    "Can you recommend a good mystery novel that keeps me guessing until the very end?",
    "I'm feeling motivated to chase my dreams and turn them into a reality.",
    "What's the best way to maintain a healthy work-life integration and find harmony?",
    "I love going on long hikes in untouched nature and disconnecting from the hustle and bustle.",
    "I'm feeling grateful for the small moments of joy that brighten my everyday life.",
    "What's the best way to overcome creative blocks and unleash my imagination?",
    "I'm trying to live a more balanced and fulfilled life, nurturing all aspects of my well-being."]

sentence_embeddings = model.encode(sentence_list)

database.prompts = sentence_list
database.embeddings = sentence_embeddings




