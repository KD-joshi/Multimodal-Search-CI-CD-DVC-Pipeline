# 🚀 Serverless Multimodal Fashion Search Engine

A production-grade **Multimodal Product Search Engine** for fashion e-commerce built with a cutting-edge hybrid cloud architecture. 

It combines Dense Semantic Retrieval using **Pinecone**, Visual Embeddings using **FashionCLIP**, and a completely serverless frontend/backend architecture split across **Vercel**, **Google Colab GPUs**, and **Kaggle** for automated MLOps.

---

## 🏗️ Architecture Overview

To achieve maximum scalability at zero cost, the architecture is split into three decoupled cloud layers:

### 1. The Frontend & Text Search Backend (Vercel)
- **Frontend**: A highly responsive, modern React application hosted on Vercel.
- **Serverless API**: Python Serverless Functions deployed on Vercel handle natural language text queries.
- **Vector Database**: **Pinecone** is used to store and retrieve dense vector embeddings for sub-10ms search latency without maintaining a server.

### 2. The Image Search Backend (Google Colab)
Running PyTorch and `fashion-clip` requires high-RAM GPUs which Vercel Serverless cannot provide.
- We utilize a **Google Colab Notebook** to spin up a free 15GB GPU instance.
- A **FastAPI** server runs on the Colab instance, instantly generating embeddings for uploaded images.
- An **Ngrok** static tunnel securely exposes the Colab GPU endpoint directly to the Vercel frontend.

### 3. Automated MLOps Retraining Pipeline (GitHub Actions 🤝 Kaggle)
To ensure the embeddings never drift as seasonal fashion trends change, an automated MLOps pipeline is established:
- **GitHub Actions** runs a CRON job every Sunday at midnight.
- It securely authenticates with the **Kaggle API** using modern `access_token` authentication.
- It remotely triggers a headless Data Drift Detection and Retraining notebook on **Kaggle's Free GPUs**, executing the heavy lifting entirely in the cloud.

---

## 💻 How to Run

### 1. Launch the Text Search & Frontend (Vercel)
1. Fork this repository.
2. Link the repository to your Vercel account.
3. Add your `PINECONE_API_KEY` to the Vercel Environment Variables.
4. Deploy! The React UI and Text Search API are instantly live.

### 2. Launch the Image Search GPU (Colab)
1. Open `colab_backend.ipynb` in Google Colab.
2. Add your `NGROK_AUTH_TOKEN`.
3. Click **Run All**.
4. The notebook will spin up FastAPI and print your public Ngrok URL.
5. Your Vercel frontend is already configured to automatically route Image Uploads to this static Colab URL!

### 3. Trigger MLOps Retraining
1. Go to your GitHub Repository -> **Actions**.
2. Click **Automated MLOps Retraining Pipeline**.
3. Click **Run Workflow**.
4. Watch GitHub remotely execute your data pipeline on Kaggle!

---

## 🛠️ Tech Stack
- **Frontend**: React, Vite, CSS
- **Backend API**: FastAPI, Vercel Serverless (Python)
- **AI Models**: Hugging Face `patrickjohncyh/fashion-clip`
- **Vector DB**: Pinecone HNSW
- **MLOps**: GitHub Actions, Kaggle Kernels API
- **Tunneling**: Ngrok Static Domains
