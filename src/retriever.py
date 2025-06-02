import os, pickle
import numpy as np
import pandas as pd
import faiss
from openai_client import openai, EMBED_ENGINE

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
IDX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")
META_PATH= os.path.join(DATA_DIR, "tickets_meta.pkl")

def load_index():
    index = faiss.read_index(IDX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta

def retrieve(query: str, top_k: int = 3):
    resp = openai.embeddings.create(engine=EMBED_ENGINE, input=[query])
    q_vec = np.array(resp["data"][0]["embedding"], dtype="float32").reshape(1, -1)

    index, meta = load_index()
    D, I = index.search(q_vec, top_k)
    results = []
    for idx in I[0]:
        row = meta.iloc[idx]
        results.append({
            "short_description": row["short_description"],
            "description":       row["description"],
            "metadata": {
                "priority":         row["priority"],
                "assignment_group": row["assignment_group"],
                "task_type":        row["task_type"],
                "root_cause_code":  row["root_cause_code"]
            }
        })
    return results

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    print(json.dumps(retrieve(args.query, args.k), indent=2))
