import os
import requests
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

def download_file_if_not_exists(json_path, download_url):
    """Check if the file exists locally; if not, download it from the specified URL."""
    if not os.path.exists(json_path):
        print(f"{json_path} not found. Downloading from {download_url}...")
        response = requests.get(download_url, stream=True)
        
        if response.status_code == 200:
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded and saved to {json_path}")
        else:
            raise Exception(f"Failed to download the file. Status code: {response.status_code}")
    else:
        print(f"File {json_path} already exists.")

def load_papers(json_path, download_url=None):
    """Load scientific papers from JSON file and return as a DataFrame. Downloads if not present."""
    if download_url:
        download_file_if_not_exists(json_path, download_url)
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    papers_data = []
    for paper in data:
        abstract = paper.get('abstractText', '')
        if abstract is None:
            continue
        co_authors = [f"{author['person']['firstname']} {author['person']['lastname']}" for author in paper.get('referenceAuthors', [])]
        papers_data.append({'abstract': abstract, 'co_authors': co_authors})

    return pd.DataFrame(papers_data)

def generate_embeddings(papers):
    """Generate sentence embeddings for paper abstracts."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    papers['embedding'] = papers['abstract'].apply(lambda x: model.encode(x))
    
    # Stack embeddings into a 2D array
    embeddings = np.vstack(papers['embedding'].values)
    
    return embeddings, papers
