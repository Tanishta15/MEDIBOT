import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    words = word_tokenize(str(text))
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

try:
    # Load the dataset with extra error handling
    file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Ensure required columns exist
    required_columns = ["Disease", "Scientific Name", "Active Component", "Focus", "Plant Name"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    # Merge relevant columns for text processing
    df['combined'] = df[['Scientific Name', 'Active Component', 'Focus', 'Plant Name']].astype(str).agg(' '.join, axis=1)
    df['processed'] = df['combined'].apply(preprocess_text)

    # LSA pipeline (TF-IDF + LSA)
    vectorizer = TfidfVectorizer()
    lsa = make_pipeline(vectorizer, TruncatedSVD(n_components=100), Normalizer(copy=False))

    # Transform the dataset into LSA space
    lsa_matrix = lsa.fit_transform(df['processed'])

    # Get user query
    query = input("Enter disease name: ").strip().lower()

    # Direct match search
    direct_match = df[df['Disease'].str.lower().str.contains(query, na=False, regex=False)]

    if not direct_match.empty:
        print("\nExact Disease Match Found:\n")
        print(direct_match[['Scientific Name', 'Active Component', 'Focus', 'Plant Name']])
    else:
        # Process the query for LSA
        query_processed = preprocess_text(query)
        query_vec = lsa.transform([query_processed])

        # Compute similarity
        similarity_scores = cosine_similarity(query_vec, lsa_matrix).flatten()

        # Retrieve best matches
        threshold = 0.1
        results = df[similarity_scores > threshold]

        if not results.empty:
            print("\nRelevant LSA Matches:\n")
            print(results[['Scientific Name', 'Active Component', 'Focus', 'Plant Name']])
        else:
            print("\nNo relevant information found for the entered disease.")

except pd.errors.ParserError as pe:
    print(f"\nCSV Parsing Error: {pe}")
    print("Possible Causes: Extra commas, incorrect delimiters, malformed lines.")
    print("Try opening the CSV in Excel and saving it again with a standard format.")

except Exception as e:
    print(f"\nAn error occurred: {e}")