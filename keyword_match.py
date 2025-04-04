import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing Function
def preprocess_text(text):
    words = word_tokenize(str(text).lower())  # Tokenize and convert to lowercase
    words = [word for word in words if word.isalnum()]  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    return ' '.join(words)

# Function to retrieve relevant plant details (excluding disease name)
def retrieve_plant_details(df, disease_name):
    if "Disease" not in df.columns:
        print("❌ Error: 'Disease' column not found in CSV.")
        return None

    # Preprocess the disease column for better matching
    df["Processed_Disease"] = df["Disease"].astype(str).apply(preprocess_text)
    disease_name = preprocess_text(disease_name)

    # Find matching rows based on disease name
    matching_rows = df[df["Processed_Disease"].str.contains(disease_name, na=False, case=False)]

    if matching_rows.empty:
        return "❌ No relevant plant details found for the given disease."

    # Drop disease column and return only relevant plant details
    return matching_rows.drop(columns=["Processed_Disease", "Disease"], errors="ignore")

# Load CSV file
file_path = "/Users/tanishta/Desktop/College/MEDIBOT/sample.csv"
df = pd.read_csv(file_path, on_bad_lines="skip")

# Get user input
disease_query = input("Enter the disease name: ")

# Retrieve and display relevant plant details
result = retrieve_plant_details(df, disease_query)

if isinstance(result, pd.DataFrame):
    print("\n🔹 Relevant Plant Details:\n", result.to_string(index=False))
else:
    print("\n", result)