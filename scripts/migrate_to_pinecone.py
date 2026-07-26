import os
import time
import numpy as np
import pandas as pd
from pinecone import Pinecone, ServerlessSpec

# Configuration
API_KEY = "pcsk_5H3Zja_DXkABHBRr6YPBTtEwG2VY9iYhe41tyL79BKWnUnVSHnNrFDQHgwMdgMibxtk6Px"
INDEX_NAME = "multimodal-fashion"
DIMENSION = 512 # Padded dimension for both text (384->512) and image (512)
BATCH_SIZE = 100

def migrate():
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=API_KEY)
    
    # 1. Create Index if it doesn't exist
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Serverless Index: {INDEX_NAME} (this takes ~1 min)...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
            print(".", end="", flush=True)
        print("\nIndex ready!")
    else:
        print(f"Index {INDEX_NAME} already exists.")
        
    index = pc.Index(INDEX_NAME)
    
    # 2. Load Data
    print("Loading local NumPy indices and Parquet data...")
    ids = np.load("../indices/product_ids.npy", allow_pickle=True)
    df = pd.read_parquet("../indices/products.parquet")
    
    text_embs = np.load("../indices/text_embeddings.npy") # (N, 384)
    img_embs = np.load("../indices/image_embeddings.npy")  # (N, 512)
    
    # Pad text embeddings to 512
    print("Zero-padding text embeddings from 384 to 512 dimensions...")
    padding = np.zeros((text_embs.shape[0], DIMENSION - text_embs.shape[1]), dtype=np.float32)
    text_embs_padded = np.hstack([text_embs, padding])
    
    # Precompute metadata dictionaries
    print("Preparing metadata...")
    metadata_list = []
    for i, row in df.iterrows():
        # Pinecone only accepts strings, numbers, booleans, or lists of strings. No NaNs/Nones.
        meta = {
            "product_name": str(row.get("product_name", "")) or "Unknown",
            "brand": str(row.get("brand", "")) or "Unknown",
            "sales_price": float(row.get("sales_price", 0.0)) if pd.notna(row.get("sales_price")) else 0.0,
            "rating": float(row.get("rating", 0.0)) if pd.notna(row.get("rating")) else 0.0,
            "image_url": str(row.get("image_url", "")) or ""
        }
        metadata_list.append(meta)
    
    total = len(ids)
    
    # 3. Upload Text Namespace
    print(f"\nUploading {total} vectors to 'text' namespace...")
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE].astype(str).tolist()
        batch_vecs = text_embs_padded[i:i+BATCH_SIZE].tolist()
        batch_meta = metadata_list[i:i+BATCH_SIZE]
        
        vectors = zip(batch_ids, batch_vecs, batch_meta)
        index.upsert(vectors=vectors, namespace="text")
        
        if (i + BATCH_SIZE) % 5000 == 0 or (i + BATCH_SIZE) >= total:
            print(f"  Uploaded {min(i+BATCH_SIZE, total)} / {total} text vectors")
            
    # 4. Upload Image Namespace
    print(f"\nUploading {total} vectors to 'image' namespace...")
    for i in range(0, total, BATCH_SIZE):
        batch_ids = ids[i:i+BATCH_SIZE].astype(str).tolist()
        batch_vecs = img_embs[i:i+BATCH_SIZE].tolist()
        batch_meta = metadata_list[i:i+BATCH_SIZE]
        
        vectors = list(zip(batch_ids, batch_vecs, batch_meta))
        index.upsert(vectors=vectors, namespace="image")
        
        if (i + BATCH_SIZE) % 5000 == 0 or (i + BATCH_SIZE) >= total:
            print(f"  Uploaded {min(i+BATCH_SIZE, total)} / {total} image vectors")
            
    print("\n✅ Migration complete! Pinecone is now fully populated.")

if __name__ == "__main__":
    migrate()
