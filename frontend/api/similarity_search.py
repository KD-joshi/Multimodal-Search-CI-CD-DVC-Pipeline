"""
Serverless Product Similarity Search Interface.

Replaces local FAISS and PyTorch with Pinecone and HuggingFace API.
"""

import os
import time
import logging
from typing import List, Dict, Optional
import numpy as np
import httpx
from pinecone import Pinecone

import config

logger = logging.getLogger(__name__)

# Serverless Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "pcsk_5H3Zja_DXkABHBRr6YPBTtEwG2VY9iYhe41tyL79BKWnUnVSHnNrFDQHgwMdgMibxtk6Px")
PINECONE_INDEX_NAME = "multimodal-fashion"
HF_TOKEN = os.getenv("HF_TOKEN", "") # Optional, but recommended for rate limits

class ProductSimilaritySearch:
    def __init__(self):
        self._initialized = False
        self.pc = None
        self.index = None
        
    def initialize(self):
        start = time.perf_counter()
        logger.info("Initializing Serverless connection to Pinecone...")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        elapsed = time.perf_counter() - start
        self._initialized = True
        logger.info(f"✅ Initialization complete in {elapsed:.1f}s")
        
    def _get_text_embedding(self, text: str) -> List[float]:
        """Call HF API for text embedding."""
        from huggingface_hub import InferenceClient
        
        # Initialize client (automatically picks up ~/.cache/huggingface/token or HF_TOKEN env var)
        client = InferenceClient(token=HF_TOKEN if HF_TOKEN else None)
        
        # This automatically routes to the correct working inference endpoint
        vector = client.feature_extraction(text, model="sentence-transformers/all-MiniLM-L6-v2")
        
        # It returns a numpy array, convert to list
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
            
        return vector
            
    def _get_image_embedding(self, image_bytes: bytes) -> List[float]:
        """Generate image embedding locally (Requires PyTorch and Colab RAM)."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            from PIL import Image
            import io
        except ImportError:
            raise RuntimeError(
                "Image search requires PyTorch and Transformers. "
                "This endpoint is disabled on the Vercel Serverless tier. "
                "Please run the backend on Google Colab to enable Image Search."
            )
            
        # Lazy load the model so it doesn't crash serverless text search
        if not hasattr(self, "clip_model"):
            logger.info("Loading FashionCLIP model into memory...")
            self.clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
            self.clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
            
        # Process image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            
        # L2 Normalize the features (Pinecone expects dotproduct metric with normalized vectors)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features[0].tolist()

    def search_by_text(self, query: str, top_k: int = 10, mode: str = "text_structured") -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
            
        # 1. Get embedding from HF
        vector = self._get_text_embedding(query)
        
        # 2. Pad to 512 dimensions for Pinecone
        padded_vector = vector + [0.0] * (512 - len(vector))
        
        # 3. Query Pinecone
        res = self.index.query(
            namespace="text",
            vector=padded_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # 4. Format results
        results = []
        for match in res.matches:
            meta = match.metadata
            results.append({
                'uniq_id': match.id,
                'product_name': meta.get('product_name'),
                'brand': meta.get('brand'),
                'sales_price': meta.get('sales_price'),
                'rating': meta.get('rating'),
                'image_url': meta.get('image_url'),
                'score': float(match.score),
            })
            
        return results

    def search_by_image(self, image_bytes: bytes, top_k: int = 10) -> List[Dict]:
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
            
        # 1. Get embedding from HF
        vector = self._get_image_embedding(image_bytes)
        
        # 2. Query Pinecone
        res = self.index.query(
            namespace="image",
            vector=vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # 3. Format results
        results = []
        for match in res.matches:
            meta = match.metadata
            results.append({
                'uniq_id': match.id,
                'product_name': meta.get('product_name'),
                'brand': meta.get('brand'),
                'sales_price': meta.get('sales_price'),
                'rating': meta.get('rating'),
                'image_url': meta.get('image_url'),
                'score': float(match.score),
            })
            
        return results

    def get_product_metadata(self, product_id: str) -> Optional[Dict]:
        """Fetch metadata for a single product directly from Pinecone."""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
            
        res = self.index.fetch(ids=[product_id], namespace="text")
        if product_id not in res.vectors:
            return None
            
        meta = res.vectors[product_id].metadata
        return {
            'uniq_id': product_id,
            'product_name': meta.get('product_name'),
            'brand': meta.get('brand'),
            'sales_price': meta.get('sales_price'),
            'rating': meta.get('rating'),
            'image_url': meta.get('image_url'),
        }

    def calculate_similarity(self, product_id: str, mode: str = "text_structured", top_k: int = 100) -> List[Dict]:
        """Find similar products to an existing product ID using Pinecone."""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
            
        namespace = "image" if mode == "image" else "text"
        
        # 1. Fetch the target product's vector
        fetch_res = self.index.fetch(ids=[product_id], namespace=namespace)
        if product_id not in fetch_res.vectors:
            raise ValueError(f"Product ID '{product_id}' not found in namespace '{namespace}'")
            
        query_vector = fetch_res.vectors[product_id].values
        
        # 2. Query Pinecone with that vector
        res = self.index.query(
            namespace=namespace,
            vector=query_vector,
            top_k=top_k + 1,  # +1 because the product itself will be returned
            include_metadata=True
        )
        
        # 3. Format results, excluding the query product
        results = []
        for match in res.matches:
            if match.id == product_id:
                continue
            meta = match.metadata
            results.append({
                'uniq_id': match.id,
                'similarity_score': float(match.score),
                'product_name': meta.get('product_name'),
                'brand': meta.get('brand'),
                'sales_price': meta.get('sales_price'),
                'rating': meta.get('rating'),
                'image_url': meta.get('image_url'),
            })
            
        return results

    def find_similar_products(self, product_id: str, num_similar: int, mode: str = "text_structured") -> List[str]:
        """Return just the IDs of similar products."""
        results = self.calculate_similarity(product_id, mode, top_k=num_similar)
        return [r['uniq_id'] for r in results]
