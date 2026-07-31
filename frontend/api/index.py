"""
System Configuration.

Centralized configuration for hyperparameters, file paths, and environment settings.
Values can be overridden via environment variables to support different environments
(development, staging, production) without requiring code modifications.
"""

import os
from pathlib import Path

# ==============================================================================
# PATHS
# ==============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = os.getenv(
    "DATA_PATH",
    str(BASE_DIR / "data" / "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
)
INDEX_DIR = os.getenv("INDEX_DIR", str(BASE_DIR / "indices"))
IMAGE_CACHE_DIR = os.getenv("IMAGE_CACHE_DIR", str(BASE_DIR / "data" / "images"))

# ==============================================================================
# FAISS HNSW PARAMETERS
# ==============================================================================
# M: bidirectional links per node. Higher = better recall, more memory.
# efConstruction: candidate list size during graph build. Higher = better graph.
# efSearch: candidate list size during query. Tunable at runtime.
HNSW_M = int(os.getenv("HNSW_M", "32"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "64"))

# ==============================================================================
# EMBEDDING MODELS
# ==============================================================================
# Text: Sentence-BERT (all-MiniLM-L6-v2), 384-dimensional embeddings.
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_EMBEDDING_DIM = 384

# Image: CLIP (ViT-B/32), 512-dimensional embeddings.
IMAGE_MODEL_NAME = os.getenv("IMAGE_MODEL_NAME", "patrickjohncyh/fashion-clip")
IMAGE_EMBEDDING_DIM = 512

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
# Keep top-N brands/colors as individual features; bucket the rest as "Other".
TOP_N_BRANDS = int(os.getenv("TOP_N_BRANDS", "100"))
TOP_N_COLORS = int(os.getenv("TOP_N_COLORS", "30"))

# Sentinel value for unknown weight in the dataset.
WEIGHT_SENTINEL = 999999999

# ==============================================================================
# SIMILARITY SEARCH
# ==============================================================================
# Late-fusion weights for combining text+structured and image scores.
TEXT_STRUCT_WEIGHT = float(os.getenv("TEXT_STRUCT_WEIGHT", "0.4"))
IMAGE_WEIGHT = float(os.getenv("IMAGE_WEIGHT", "0.6"))

# Number of candidates to send to the Cross-Encoder for Stage-2 reranking.
# Higher = better accuracy but slower. Lower = much faster.
RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "15"))

# ==============================================================================
# CACHING
# ==============================================================================
# LRU cache size for find_similar_products results.
LRU_CACHE_SIZE = int(os.getenv("LRU_CACHE_SIZE", "1024"))

# ==============================================================================
# DIMENSIONALITY REDUCTION
# ==============================================================================
# Optional PCA compression of the combined feature vector.
USE_PCA = os.getenv("USE_PCA", "false").lower() == "true"
PCA_COMPONENTS = int(os.getenv("PCA_COMPONENTS", "256"))

# ==============================================================================
# API
# ==============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))


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


"""
FastAPI Microservice for Product Similarity Search (Serverless Edition).

End-to-End completely free serverless API architecture.
Loads zero vectors locally, uses Pinecone for storage and HuggingFace API for embeddings.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)
logger = logging.getLogger(__name__)

# Supabase Initialization for MLOps Logging
import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Connected to Supabase for Search Logging.")
    except Exception as e:
        logger.error(f"Failed to connect to Supabase: {e}")
else:
    logger.warning("⚠️ SUPABASE_URL or SUPABASE_KEY not found. Search logging will be disabled.")

def log_search_query(query_text: str, search_mode: str, latency_ms: float, results_count: int):
    """Background task to log user queries to Supabase for Data Drift Detection."""
    if not supabase:
        return
        
    try:
        data = {
            "query_text": query_text,
            "search_mode": search_mode,
            "latency_ms": latency_ms,
            "results_count": results_count
        }
        supabase.table("search_logs").insert(data).execute()
        logger.info(f"📊 Logged {search_mode} search to Supabase.")
    except Exception as e:
        logger.error(f"Failed to log search: {e}")

# Global search instance
search: Optional[ProductSimilaritySearch] = None
purged_ids: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    Initializes connection to Pinecone Vector DB.
    """
    global search
    logger.info("🚀 Starting Serverless Search API...")
    
    search = ProductSimilaritySearch()
    search.initialize()
    
    logger.info(f"✅ Serverless API ready! Connected to Pinecone.")
    
    yield
    
    logger.info("👋 Shutting down API...")
    search = None


app = FastAPI(
    title="Multimodal Fashion Search API (Serverless)",
    description="True serverless search using Pinecone and HuggingFace API.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: str = "text_structured"


@app.post("/api/search/text")
async def search_text(req: TextSearchRequest, background_tasks: BackgroundTasks):
    """
    Search using natural language.
    Passes query through HuggingFace Inference API, then queries Pinecone.
    """
    if not search:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
        
    try:
        start_time = time.time()
        results = search.search_by_text(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode
        )
        latency = (time.time() - start_time) * 1000
        
        # Log to Supabase for Drift Detection
        background_tasks.add_task(log_search_query, req.query, "text", latency, len(results))
        
        return {
            "results": results,
            "query_type": "text",
            "query": req.query,
            "latency_ms": round(latency, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Text search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/image")
async def search_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    image_url: str = Form(None),
    top_k: int = Form(10)
):
    """
    Search using an uploaded image.
    Requires PyTorch locally (Will fail gracefully if on Serverless).
    """
    if not search:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
        
    try:
        start_time = time.time()
        image_bytes = await file.read() if file else None
        results = search.search_by_image(
            image_bytes=image_bytes,
            top_k=top_k
        )
        latency = (time.time() - start_time) * 1000
        
        # Log to Supabase for Drift Detection
        background_tasks.add_task(log_search_query, f"[Image Upload: {file.filename if file else image_url}]", "image", latency, len(results))
        
        return {
            "results": results,
            "query_type": "image",
            "latency_ms": round(latency, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Image search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health_check():
    if search is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "healthy"}


@app.get("/api/find_similar_products")
def get_similar_products(
    product_id: str = Query(...),
    num_similar: int = Query(...),
    mode: str = Query("text_structured"),
) -> List[str]:
    if search is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        start = time.perf_counter()
        similar_products = search.find_similar_products(product_id, num_similar, mode=mode)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return similar_products
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/product/{product_id}")
def get_product_details(product_id: str):
    if search is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    meta = search.get_product_metadata(product_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Product ID not found")
        
    return meta


@app.get("/api/find_similar_products_detailed")
def get_similar_products_detailed(
    product_id: str = Query(...),
    num_similar: int = Query(10, gt=0),
    mode: str = Query("text_structured"),
):
    if search is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        query_row = search.get_product_metadata(product_id)
        if not query_row:
            raise HTTPException(status_code=404, detail="Product ID not found")
            
        dict_results = search.calculate_similarity(product_id, mode=mode, top_k=num_similar)
        
        return {
            "query_product": query_row,
            "similar_products": dict_results[:num_similar],
            "mode": mode,
        }
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
