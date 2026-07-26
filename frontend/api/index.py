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

from similarity_search import ProductSimilaritySearch
import config

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
            image_url=image_url,
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
