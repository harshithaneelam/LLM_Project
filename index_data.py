import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("synthetic_dataset.csv")

# Initialize the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text data into embeddings
embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True)

# Create a FAISS index
dimension = embeddings.shape[1]  # Embedding size
index = faiss.IndexFlatL2(dimension)  # L2 distance index
index.add(embeddings)  # Add embeddings to the index

# Save the FAISS index
faiss.write_index(index, "faiss_index.bin")
np.save("metadata.npy", df.to_numpy(), allow_pickle=True)

print("FAISS index created and saved as 'faiss_index.bin'. Metadata saved as 'metadata.npy'.")
 
