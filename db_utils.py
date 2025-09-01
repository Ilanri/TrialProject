import os
import pickle
import faiss
import numpy as np
import logging

DEBUG = os.environ.get("DEBUG", "0") == "1"
logger = logging.getLogger("db_utils")

def save_faiss_index(index, path):
    try:
        logger.debug(f"Saving FAISS index to {path}")
        faiss.write_index(index, path)
        logger.info(f"Saved FAISS index to {path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index to {path}: {e}")
        raise

def load_faiss_index(path):
    try:
        logger.debug(f"Loading FAISS index from {path}")
        index = faiss.read_index(path)
        logger.info(f"Loaded FAISS index from {path}")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {path}: {e}")
        raise

def save_docs_and_embeddings(docs, embeddings, path):
    try:
        logger.debug(f"Saving docs and embeddings to {path}")
        with open(path, 'wb') as f:
            pickle.dump({'docs': docs, 'embeddings': embeddings}, f)
        logger.info(f"Saved docs and embeddings to {path}")
    except Exception as e:
        logger.error(f"Failed to save docs and embeddings to {path}: {e}")
        raise

def load_docs_and_embeddings(path):
    try:
        logger.debug(f"Loading docs and embeddings from {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        docs, embeddings = data['docs'], data['embeddings']
        logger.info(f"Loaded docs and embeddings from {path}")
        return docs, embeddings
    except Exception as e:
        logger.error(f"Failed to load docs and embeddings from {path}: {e}")
        raise
