import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to find the closest disease using BERT similarity
def find_closest_disease(input_disease, disease_list):
    disease_embeddings = model.encode(disease_list, convert_to_tensor=True)
    input_embedding = model.encode(input_disease, convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(input_embedding, disease_embeddings)[0]
    best_match_idx = torch.argmax(similarities).item()
    
    # Use a threshold to avoid incorrect matches
    if similarities[best_match_idx] > 0.5:
        return disease_list[best_match_idx]
    return None

try:
    # Load CSV file
    file_path = '/Users/tanishta/Desktop/College/MEDIBOT/sample.csv'
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Ensure required columns exist
    required_columns = {"Disease", "Plant Name", "Scientific Name", "Active Component", "Focus"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns in CSV. Expected columns: {required_columns}")

    # Get user input for disease
    input_disease = input("Enter the disease name: ").strip()

    # Check if input disease exists; otherwise, find the closest match
    disease_list = df["Disease"].dropna().unique().tolist()
    if input_disease not in disease_list:
        closest_disease = find_closest_disease(input_disease, disease_list)
        if closest_disease:
            print(f"Did you mean: {closest_disease}? (Using closest match)")
            input_disease = closest_disease
        else:
            print("No similar disease found. Please check your input.")
            exit()

    # Retrieve relevant plant information
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