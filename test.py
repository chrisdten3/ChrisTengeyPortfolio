import pandas as pd
import re
from gensim.utils import simple_preprocess

# Load your DataFrame (replace 'your_dataframe.csv' with your actual file)
df = pd.read_csv('your_dataframe.csv')

# Function to clean and tokenize text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text
    tokens = simple_preprocess(text)
    return tokens

# Apply preprocessing to relevant columns
df['subprocess_name_tokens'] = df['subprocess_name'].apply(preprocess_text)
df['subprocess_description_tokens'] = df['subprocess_description'].apply(preprocess_text)
df['risk_type_tokens'] = df['risk_type'].apply(preprocess_text)
df['rationale_tokens'] = df['rationale'].apply(preprocess_text)
df['controls_description_tokens'] = df['controls_description'].apply(preprocess_text)


import numpy as np

def get_average_vector(tokens, model):
    # Filter tokens that are in the model's vocabulary
    tokens = [token for token in tokens if token in model.wv.key_to_index]
    if not tokens:
        return np.zeros(model.vector_size)
    # Calculate the average vector
    vectors = [model.wv[token] for token in tokens]
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector

# Apply vectorization to relevant columns
df['subprocess_name_vector'] = df['subprocess_name_tokens'].apply(lambda x: get_average_vector(x, model))
df['subprocess_description_vector'] = df['subprocess_description_tokens'].apply(lambda x: get_average_vector(x, model))
df['risk_type_vector'] = df['risk_type_tokens'].apply(lambda x: get_average_vector(x, model))
df['rationale_vector'] = df['rationale_tokens'].apply(lambda x: get_average_vector(x, model))
df['controls_description_vector'] = df['controls_description_tokens'].apply(lambda x: get_average_vector(x, model))


from sklearn.metrics.pairwise import cosine_similarity

# Define a function to calculate similarity between two rows
def calculate_similarity(row1, row2):
    # Combine vectors into one for each row
    vector1 = np.hstack([
        row1['subprocess_name_vector'],
        row1['subprocess_description_vector'],
        row1['risk_type_vector'],
        row1['rationale_vector'],
        row1['controls_description_vector']
    ])
    vector2 = np.hstack([
        row2['subprocess_name_vector'],
        row2['subprocess_description_vector'],
        row2['risk_type_vector'],
        row2['rationale_vector'],
        row2['controls_description_vector']
    ])
    # Calculate cosine similarity
    similarity = cosine_similarity([vector1], [vector2])[0][0]
    return similarity

# Find potential duplicates by comparing each row with every other row
duplicate_pairs = []
similarity_threshold = 0.8  # Set a threshold for duplicates

for i in range(len(df)):
    for j in range(i+1, len(df)):
        similarity = calculate_similarity(df.iloc[i], df.iloc[j])
        if similarity > similarity_threshold:
            duplicate_pairs.append((i, j, similarity))

# Print potential duplicates
for pair in duplicate_pairs:
    print(f"Duplicate pair: Row {pair[0]} and Row {pair[1]} with similarity {pair[2]}")

#can also use annoy
from annoy import AnnoyIndex

# Define the dimensionality of the vectors
vector_size = 100

# Create an Annoy index
annoy_index = AnnoyIndex(vector_size, 'angular')

# Add vectors to the index
for i, vector in enumerate(df['subprocess_name_vector']):
    annoy_index.add_item(i, vector)

# Build the index (can set n_trees for better accuracy)
annoy_index.build(10)

# Function to find duplicates
def find_duplicates(index, df, threshold=0.8):
    duplicate_pairs = []
    for i in range(len(df)):
        neighbors = index.get_nns_by_item(i, 10, include_distances=True)
        for j, distance in zip(neighbors[0], neighbors[1]):
            if i != j and distance < (1 - threshold):
                duplicate_pairs.append((i, j, 1 - distance))
    return duplicate_pairs

# Find duplicates with a threshold of 0.8
duplicates = find_duplicates(annoy_index, df, threshold=0.8)

# Print potential duplicates
for pair in duplicates:
    print(f"Duplicate pair: Row {pair[0]} and Row {pair[1]} with similarity {pair[2]}")

