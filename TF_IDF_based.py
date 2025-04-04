import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(str(text))
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Function to retrieve relevant rows based on TF-IDF similarity
def retrieve_by_tfidf(df, query, threshold=0.2):
    diseases = df['Disease'].fillna('').astype(str).tolist()
    processed_diseases = [preprocess_text(d) for d in diseases]
    processed_query = preprocess_text(query)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_diseases)
    query_vector = vectorizer.transform([processed_query])

    similarities = np.dot(tfidf_matrix, query_vector.T).toarray().flatten()

    relevant_indices = [i for i, sim in enumerate(similarities) if sim >= threshold]

    results = []
    for i in relevant_indices:
        row = df.iloc[i]
        results.append({
            'Scientific Name': row.get('Scientific Name', 'Unknown'),
            'Active Component': row.get('Active Component', 'Unknown'),
            'Focus': row.get('Focus', 'Unknown'),
            'Plant Name': row.get('Plant Name', 'Unknown')
        })

    return results

# Main script
file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'

try:
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Check necessary columns
    required_cols = ['Scientific Name', 'Active Component', 'Focus', 'Plant Name', 'Disease']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("One or more required columns are missing from the dataset.")

    # Get user input
    query = input("Enter the disease name: ")

    results = retrieve_by_tfidf(df, query)

    if results:
        print("\nRelevant Matches:")
        for entry in results:
            print(entry)
    else:
        print("\nNo relevant information found on the disease.")

except FileNotFoundError:
    print("\nError: File not found. Please check the path.")