import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function - cleans and standardizes text
def preprocess_text(text):
    words = word_tokenize(str(text))  # Ensure text is a string
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Function to find closest disease using BM25
def find_closest_disease(input_disease, disease_list):
    tokenized_diseases = [preprocess_text(disease) for disease in disease_list]
    bm25 = BM25Okapi(tokenized_diseases)
    
    processed_input = preprocess_text(input_disease)
    scores = bm25.get_scores(processed_input)
    
    best_match_idx = scores.argmax()
    best_match = disease_list[best_match_idx] if scores[best_match_idx] > 0.1 else None  # Adjust threshold as needed
    return best_match

try:
    # Load CSV file
    file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Ensure required columns exist
    required_columns = {"Disease", "Plant Name", "Scientific Name", "Active Component", "Focus"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected columns: {required_columns}")

    # Get user input for disease
    input_disease = input("Enter the disease: ").strip()

    # Check if input disease exists in dataset, otherwise find the closest match
    disease_list = df["Disease"].dropna().unique()
    if input_disease not in disease_list:
        closest_disease = find_closest_disease(input_disease, disease_list)
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

except FileNotFoundError:
    print("\nError: File not found. Please check the file path and try again.")
except pd.errors.ParserError as pe:
    print(f"\nCSV Parsing Error: {pe}")
except Exception as e:
    print(f"\nAn error occurred: {e}")