import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load pre-trained FastText vectors
fasttext.util.download_model('en', if_exists='ignore')  # English
ft_model = fasttext.load_model('cc.en.300.bin')

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(str(text))
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Train Word2Vec model
def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Get vector for a given text
def get_text_vector(text, model):
    words = preprocess_text(text)
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Find the closest disease name using similarity
def find_closest_disease(input_disease, disease_list, model):
    input_vector = get_text_vector(input_disease, model)
    best_match = None
    highest_similarity = -1

    for disease in disease_list:
        disease_vector = get_text_vector(disease, model)
        if np.linalg.norm(input_vector) == 0 or np.linalg.norm(disease_vector) == 0:
            continue  # Skip empty vectors
        similarity = np.dot(input_vector, disease_vector) / (np.linalg.norm(input_vector) * np.linalg.norm(disease_vector))
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = disease

    return best_match if highest_similarity > 0.5 else None  # Threshold for similarity

try:
    # Load the CSV file
    file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Ensure required columns exist
    required_columns = {"Disease", "Plant Name", "Scientific Name", "Active Component", "Focus"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected columns: {required_columns}")

    # Train Word2Vec model on all text data
    processed_sentences = [preprocess_text(str(text)) for text in df["Disease"].dropna()]
    model = train_word2vec(processed_sentences)

    # User input for disease
    input_disease = input("Enter the disease: ").strip()

    # Check if input disease exists in dataset, otherwise find the closest match
    if input_disease not in df["Disease"].values:
        closest_disease = find_closest_disease(input_disease, df["Disease"].dropna().values, model)
        if closest_disease:
            print(f"Did you mean: {closest_disease}? (Using closest match)")
            input_disease = closest_disease
        else:
            print("No similar disease found. Please check your input.")
            exit()

    # Retrieve relevant information for the disease
    result_df = df[df["Disease"] == input_disease][["Plant Name", "Scientific Name", "Active Component", "Focus"]]

    if not result_df.empty:
        print("\nRelevant Medicinal Plants for the Disease:\n")
        print(result_df.to_string(index=False))
    else:
        print("\nNo matching plants found for the disease.")

except pd.errors.ParserError as pe:
    print(f"\nCSV Parsing Error: {pe}")
    print("Possible Causes: Extra commas, incorrect delimiters, malformed lines.")

except Exception as e:
    print(f"\nAn error occurred: {e}")