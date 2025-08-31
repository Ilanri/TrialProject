import os
import pickle
import faiss
import numpy as np

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def save_docs_and_embeddings(docs, embeddings, path):
    with open(path, 'wb') as f:
        pickle.dump({'docs': docs, 'embeddings': embeddings}, f)

def load_docs_and_embeddings(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['docs'], data['embeddings']
