import os
import json
import numpy as np

DB_DIR = "data/identity_db"
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR, "labels.json")


def load_db():
    if not os.path.exists(EMB_PATH) or os.path.getsize(EMB_PATH) == 0:
        return np.empty((0, 512)), {}

    try:
        embeddings = np.load(EMB_PATH)
    except Exception:
        return np.empty((0, 512)), {}

    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            labels = json.load(f)
    else:
        labels = {}

    return embeddings, labels


def save_db(embeddings, labels):
    os.makedirs(DB_DIR, exist_ok=True)
    np.save(EMB_PATH, embeddings)

    with open(LABEL_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def add_identity(name, embedding):
    embeddings, labels = load_db()

    embeddings = np.vstack([embeddings, embedding])
    labels[str(len(labels))] = name

    save_db(embeddings, labels)

def add_identity_with_embedding(name, embedding):
    embeddings, labels = load_db()

    if name in labels.values():
        raise ValueError(f"Identity '{name}' already exists")

    embeddings = np.vstack([embeddings, embedding])
    labels[str(len(labels))] = name

    save_db(embeddings, labels)

