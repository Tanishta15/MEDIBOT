import nltk
import requests
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall_score
from nltk.tokenize import word_tokenize
import numpy as np
from io import StringIO

# Download NLTK resources
nltk.download('punkt')

# Load data from GitHub
url = 'https://github.com/FeiYee/HerbKG/raw/main/DiseaseData/test.tsv'
response = requests.get(url)
data = pd.read_csv(StringIO(response.text), sep='\t')

# Assume the file has 'text' and 'label' columns
texts = data['text']
labels = data['label']

# Pre-trained Word2Vec model (use a smaller model for simplicity)
word2vec_model_path = 'GoogleNews-vectors-negative300.bin.gz'  # Example path
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

# Predefined keywords
positive_keywords = {'cure', 'heal', 'improve'}
negative_keywords = {'harm', 'worsen', 'negative'}

# Preprocess text
def preprocess_text(text):
    return word_tokenize(text.lower())

# Word2Vec predictions
def word2vec_predict(text):
    tokens = preprocess_text(text)
    vectors = [word2vec_model[word] for word in tokens if word in word2vec_model]
    if not vectors:
        return "neutral"  # No known words
    average_vector = np.mean(vectors, axis=0)
    return "positive" if average_vector.mean() > 0 else "negative"

# Key matching predictions
def key_matching_predict(text):
    tokens = set(preprocess_text(text))
    if tokens & positive_keywords:
        return "positive"
    elif tokens & negative_keywords:
        return "negative"
    return "neutral"

# Evaluate models
word2vec_predictions = [word2vec_predict(text) for text in texts]
key_matching_predictions = [key_matching_predict(text) for text in texts]

# Calculate evaluation metrics
def evaluate(true_labels, predictions):
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    return precision, accuracy, f1, recall

word2vec_metrics = evaluate(labels, word2vec_predictions)
key_matching_metrics = evaluate(labels, key_matching_predictions)

# Print results
print("Word2Vec Metrics:")
print(f"Precision: {word2vec_metrics[0]:.2f}, Accuracy: {word2vec_metrics[1]:.2f}, F1 Score: {word2vec_metrics[2]:.2f}, Recall: {word2vec_metrics[3]:.2f}")

print("\nKey Matching Metrics:")
print(f"Precision: {key_matching_metrics[0]:.2f}, Accuracy: {key_matching_metrics[1]:.2f}, F1 Score: {key_matching_metrics[2]:.2f}, Recall: {key_matching_metrics[3]:.2f}")
