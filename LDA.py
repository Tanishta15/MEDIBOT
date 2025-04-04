import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
import torch
from sentence_transformers import SentenceTransformer, util

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load Sentence Transformer for similarity checking
bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Text Preprocessing Function
def preprocess_text(text):
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Function to train LDA model
def train_lda(sentences, num_topics=5):
    processed_sentences = [preprocess_text(sentence) for sentence in sentences]
    dictionary = corpora.Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(text) for text in processed_sentences]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
    return lda_model, dictionary, corpus, processed_sentences

# Function to find the closest matching disease
def find_closest_disease(input_disease, disease_list):
    disease_embeddings = bert_model.encode(disease_list, convert_to_tensor=True)
    input_embedding = bert_model.encode(input_disease, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(input_embedding, disease_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()

    if similarities[best_match_idx] > 0.5:  # Threshold for correction
        return disease_list[best_match_idx]
    return None

# Function to retrieve relevant information using LDA
def retrieve_information(lda_model, dictionary, corpus, sentences, topic):
    topic_bow = dictionary.doc2bow(preprocess_text(topic))
    topic_distribution = lda_model[topic_bow]

    topic_scores = []
    for i, doc in enumerate(corpus):
        doc_distribution = lda_model[doc]
        score = sum(min(doc_prob, topic_prob) for (doc_topic, doc_prob) in doc_distribution for (topic_topic, topic_prob) in topic_distribution if doc_topic == topic_topic)
        topic_scores.append((i, score))

    topic_scores.sort(key=lambda x: x[1], reverse=True)
    relevant_sentences = [sentences[i] for i, score in topic_scores if score > 0.05]
    return relevant_sentences

try:
    # Load CSV file
    file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Ensure required columns exist
    required_columns = {"Disease", "Plant Name", "Scientific Name", "Active Component", "Focus"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected columns: {required_columns}")

    # Get user input
    input_disease = input("Enter the disease name: ").strip()

    # Check for spelling correction
    disease_list = df["Disease"].dropna().unique().tolist()
    if input_disease not in disease_list:
        closest_disease = find_closest_disease(input_disease, disease_list)
        if closest_disease:
            print(f"Did you mean: {closest_disease}? (Using closest match)")
            input_disease = closest_disease
        else:
            print("No similar disease found. Please check your input.")
            exit()

    # Retrieve structured plant information
    result_df = df[df["Disease"] == input_disease][["Plant Name", "Scientific Name", "Active Component", "Focus"]]

    if not result_df.empty:
        print("\nRelevant Medicinal Plants for the Disease:\n")
        print(result_df.to_string(index=False))
    else:
        print("\nNo matching plants found for the disease.")

except FileNotFoundError:
    print("\nError: File not found. Please check the file path and try again.")
except pd.errors.ParserError as pe:
    print(f"\nCSV Parsing Error: {pe}\nPossible Causes: Extra commas, incorrect delimiters, malformed lines.")
except Exception as e:
    print(f"\nAn error occurred: {e}")