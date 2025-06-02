import os
import pickle
import numpy as np
import pandas as pd
import faiss

from dotenv import load_dotenv
from openai_client import client, EMBED_ENGINE

# Load .env (in case it hasn’t been)
load_dotenv()

# Paths
BASE_DIR     = os.path.dirname(__file__)
DATA_DIR     = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
TICKETS_PATH = os.path.join(DATA_DIR, "tickets.parquet")
INDEX_PATH   = os.path.join(DATA_DIR, "faiss_index.idx")
META_PATH    = os.path.join(DATA_DIR, "tickets_meta.pkl")

def load_tickets() -> pd.DataFrame:
    if not os.path.exists(TICKETS_PATH):
        raise FileNotFoundError(f"{TICKETS_PATH} not found. Run ingestion first.")
    return pd.read_parquet(TICKETS_PATH)

def embed_documents(batch_size: int = 20):
    df = load_tickets()

    # Prepare text blobs
    texts = (
        df["short_description"].fillna("").astype(str) +
        "\n" +
        df["description"].fillna("").astype(str)
    ).tolist()

    embeddings = []
    # Use the client.embeddings.create(...) with deployment_id
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp  = client.embeddings.create(
            deployment_id=EMBED_ENGINE,
            input=batch
        )
        embeddings.extend([record.embedding for record in resp.data])

    # Build FAISS index
    dim   = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))
    faiss.write_index(index, INDEX_PATH)
    print(f"[✓] FAISS index saved to {INDEX_PATH} ({len(embeddings)} vectors)")

    # Save metadata
    meta_df = df[[
        "short_description",
        "description",
        "priority",
        "assignment_group",
        "task_type",
        "root_cause_code"
    ]].copy()
    with open(META_PATH, "wb") as f:
        pickle.dump(meta_df, f)
    print(f"[✓] Metadata saved to {META_PATH}")

if __name__ == "__main__":
    embed_documents()
