import os
import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')


@app.get("/embed")
def read_root(sentence: str):
    embedding = model.encode(sentence, convert_to_tensor=True)
    # Convert to list for JSON serialization
    return {"embedding": embedding.tolist()}
