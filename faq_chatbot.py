import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

faqs = [
    ("What is Artificial Intelligence?",
     "AI enables computers to perform tasks that normally require human intelligence."),
    ("What is Machine Learning?",
     "Machine Learning allows computers to learn from data and make predictions."),
    ("What is Deep Learning?",
     "Deep Learning uses neural networks with multiple layers."),
    ("Where is AI used?",
     "AI is used in healthcare, education, banking, transportation, and customer service."),
    ("What are the benefits of AI?",
     "AI automates tasks, analyzes data, and improves efficiency."),
    ("What is NLP?",
     "NLP stands for Natural Language Processing. It helps computers understand human language."),
    ("What is a chatbot?",
     "A chatbot communicates with users and answers their questions."),
    ("Is AI useful in education?",
     "Yes. AI helps students learn and find answers to questions.")
]

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(nltk.word_tokenize(text))

questions = [q for q, a in faqs]
vectorizer = TfidfVectorizer(preprocessor=preprocess)
faq_vectors = vectorizer.fit_transform(questions)

def chatbot(user_question):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    best_index = similarities.argmax()

    if similarities[best_index] < 0.15:
        return "Sorry, I don't know the answer to that question."

    return faqs[best_index][1]

if __name__ == "__main__":
    print("=== AI FAQ CHATBOT ===")
    while True:
        question = input("Ask a question (type 'exit' to stop): ")
        if question.strip().lower() == "exit":
            print("Goodbye!")
            break
        print("Chatbot:", chatbot(question))
