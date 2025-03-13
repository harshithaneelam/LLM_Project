from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load FAISS index and metadata
try:
    index = faiss.read_index("faiss_index.bin")
    metadata = np.load("metadata.npy", allow_pickle=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Error loading model or FAISS index: {e}")

@app.get("/search/")
def search(query: str = Query(..., title="Search Query"), top_k: int = 5):
    """
    Perform a similarity search using FAISS.
    """
    try:
        # Convert query to embedding
        query_embedding = model.encode([query], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = index.search(query_embedding, top_k)

        # Retrieve the matching results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue  # Ignore invalid results
            results.append({
                "rank": i + 1,
                "text": metadata[idx][0],  # Assuming the text column is the first column
                "distance": float(distances[0][i])
            })

        return {"query": query, "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


