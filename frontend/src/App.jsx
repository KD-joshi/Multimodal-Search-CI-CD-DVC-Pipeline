import React, { useState, useRef } from 'react';
import { Search, Image as ImageIcon, Hash, UploadCloud, ChevronRight, Link as LinkIcon } from 'lucide-react';
import './App.css';

// Default to local, but allow overriding via localStorage
const API_BASE_URL = localStorage.getItem('API_BASE_URL') || 'https://ahoy-september-relocate.ngrok-free.dev';

function App() {
  const [activeMode, setActiveMode] = useState('text');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [meta, setMeta] = useState(null);

  // Form State
  const [textQuery, setTextQuery] = useState('');
  const [textTopK, setTextTopK] = useState(10);

  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [imageTopK, setImageTopK] = useState(10);

  const [uidQuery, setUidQuery] = useState('ad8a5a196d515ef09dfdaf082bdc37c4');
  const [uidMode, setUidMode] = useState('image');
  const [uidTopK, setUidTopK] = useState(10);

  const fileInputRef = useRef(null);

  const handleSearchText = async () => {
    if (!textQuery.trim()) return;
    setLoading(true); setError(null); setResults(null);
    try {
      const res = await fetch(`${API_BASE_URL}/search/text`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        },
        body: JSON.stringify({ query: textQuery, top_k: textTopK })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server Error');
      setResults(data.results);
      setMeta({ queryDesc: `Text: "${textQuery}"`, count: data.results.length, latency: data.latency_ms });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchImage = async () => {
    if (!imageFile && !imageUrl.trim()) return;
    setLoading(true); setError(null); setResults(null);

    try {
      let formData = new FormData();
      formData.append('top_k', imageTopK);

      if (imageFile) {
        formData.append('file', imageFile);
      } else if (imageUrl) {
        // Download the image client-side first
        const imgRes = await fetch(imageUrl);
        if (!imgRes.ok) throw new Error("Failed to fetch image from URL (might be CORS blocked).");
        const blob = await imgRes.blob();
        formData.append('file', blob, 'query.jpg');
      }

      const res = await fetch(`${API_BASE_URL}/search/image`, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': 'true'
        },
        body: formData
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server Error');
      setResults(data.results);
      setMeta({ queryDesc: `Image Upload`, count: data.results.length, latency: data.latency_ms });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchUid = async () => {
    if (!uidQuery.trim()) return;
    setLoading(true); setError(null); setResults(null);
    try {
      const res = await fetch(`${API_BASE_URL}/find_similar_products_detailed?product_id=${uidQuery}&num_similar=${uidTopK}&mode=${uidMode}`, {
        headers: {
          'ngrok-skip-browser-warning': 'true'
        }
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Server Error');

      const formattedResults = data.similar_products.map(p => ({
        ...p,
        score: p.similarity_score
      }));
      setResults(formattedResults);
      setMeta({ queryDesc: `ID: ${uidQuery.slice(0, 8)}... (${uidMode})`, count: formattedResults.length, latency: null });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="bg-blobs">
        <div className="blob blob-1"></div>
        <div className="blob blob-2"></div>
      </div>

      <div className="app-wrapper">
        <div className="container">

          <header className="header-section animate-fade-in">
            <h1 className="display-title">Multimodal Search</h1>
            <p className="header-desc">
              Discover products through natural language, visual uploads, or ID queries. Powered by Sentence-BERT and FashionCLIP.
            </p>
          </header>

          <div className="tabs-container animate-fade-in" style={{ animationDelay: '100ms' }}>
            <button className={`tab-btn ${activeMode === 'text' ? 'active' : ''}`} onClick={() => setActiveMode('text')}>
              <Search size={16} /> Text
            </button>
            <button className={`tab-btn ${activeMode === 'image' ? 'active' : ''}`} onClick={() => setActiveMode('image')}>
              <ImageIcon size={16} /> Image
            </button>
            <button className={`tab-btn ${activeMode === 'uid' ? 'active' : ''}`} onClick={() => setActiveMode('uid')}>
              <Hash size={16} /> Product ID
            </button>
          </div>

          {/* Search Panels */}
          <div className="animate-fade-in" style={{ animationDelay: '200ms' }}>
            {activeMode === 'text' && (
              <div className="search-panel">
                <div className="search-row">
                  <div className="input-group">
                    <Search className="input-icon" size={20} />
                    <input
                      type="text"
                      className="search-input"
                      placeholder="Try: 'men casual shirt blue'"
                      value={textQuery}
                      onChange={e => setTextQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSearchText()}
                    />
                  </div>
                  <select className="select-input" value={textTopK} onChange={e => setTextTopK(Number(e.target.value))}>
                    <option value={5}>Top 5</option>
                    <option value={10}>Top 10</option>
                    <option value={20}>Top 20</option>
                  </select>
                  <button className="btn-primary" onClick={handleSearchText} disabled={!textQuery.trim()}>
                    Search <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            )}

            {activeMode === 'image' && (
              <div className="search-panel">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                  {/* File Upload Row */}
                  <div className="search-row">
                    <div
                      className="file-upload-wrapper"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <UploadCloud className="upload-icon" size={32} />
                      <div className="upload-text">
                        {imageFile ? imageFile.name : 'Click or drag image to upload'}
                      </div>
                      <div className="upload-subtext">Supports JPG, PNG, WEBP</div>
                      <input
                        type="file"
                        ref={fileInputRef}
                        accept="image/*"
                        onChange={e => {
                          if (e.target.files && e.target.files[0]) {
                            setImageFile(e.target.files[0]);
                            setImageUrl(''); // Clear URL if file selected
                          }
                        }}
                      />
                    </div>
                  </div>

                  <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.875rem', fontWeight: 600 }}>OR</div>

                  {/* URL Input Row */}
                  <div className="search-row">
                    <div className="input-group">
                      <LinkIcon className="input-icon" size={20} />
                      <input
                        type="text"
                        className="search-input"
                        placeholder="Paste an image URL..."
                        value={imageUrl}
                        onChange={e => {
                          setImageUrl(e.target.value);
                          if (e.target.value) setImageFile(null); // Clear file if URL entered
                        }}
                        onKeyDown={e => e.key === 'Enter' && handleSearchImage()}
                      />
                    </div>
                    <select className="select-input" value={imageTopK} onChange={e => setImageTopK(Number(e.target.value))}>
                      <option value={5}>Top 5</option>
                      <option value={10}>Top 10</option>
                      <option value={20}>Top 20</option>
                    </select>
                    <button className="btn-primary" onClick={handleSearchImage} disabled={!imageFile && !imageUrl.trim()}>
                      Search <ChevronRight size={16} />
                    </button>
                  </div>
                </div>
              </div>
            )}

            {activeMode === 'uid' && (
              <div className="search-panel">
                <div className="search-row">
                  <div className="input-group">
                    <Hash className="input-icon" size={20} />
                    <input
                      type="text"
                      className="search-input"
                      placeholder="Enter Product UID..."
                      value={uidQuery}
                      onChange={e => setUidQuery(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && handleSearchUid()}
                    />
                  </div>
                  <select className="select-input" value={uidMode} onChange={e => setUidMode(e.target.value)}>
                    <option value="image">Image Only (CLIP)</option>
                    <option value="text_structured">Text Only (SBERT)</option>
                  </select>
                  <select className="select-input" value={uidTopK} onChange={e => setUidTopK(Number(e.target.value))}>
                    <option value={5}>Top 5</option>
                    <option value={10}>Top 10</option>
                    <option value={20}>Top 20</option>
                  </select>
                  <button className="btn-primary" onClick={handleSearchUid} disabled={!uidQuery.trim()}>
                    Search <ChevronRight size={16} />
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div style={{ color: '#EF4444', textAlign: 'center', margin: '20px 0', fontFamily: 'monospace' }}>
              Error: {error}
            </div>
          )}

          {/* Loader */}
          {loading && (
            <div className="loader-container">
              <div className="spinner"></div>
              <div className="loader-text">Querying Vector Graph</div>
            </div>
          )}

          {/* Results */}
          {!loading && results && (
            <div className="animate-fade-in">
              <div className="results-header">
                <h2 className="results-title">Results</h2>
                {meta && (
                  <div className="results-meta">
                    <div className="meta-chip">Query: <span>{meta.queryDesc}</span></div>
                    <div className="meta-chip">Count: <span>{meta.count}</span></div>
                    {meta.latency != null && (
                      <div className="meta-chip">Latency: <span>{meta.latency.toFixed(1)}ms</span></div>
                    )}
                  </div>
                )}
              </div>

              <div className="product-grid">
                {results.map((product, idx) => (
                  <div key={product.uniq_id} className="product-card" style={{ animationDelay: `${idx * 50}ms` }}>
                    <div className="rank-badge">{idx + 1}</div>
                    <div className="image-container">
                      <img
                        src={product.image_url || 'https://via.placeholder.com/400x400?text=No+Image'}
                        alt={product.product_name}
                        className="product-image"
                        onError={(e) => { e.target.onerror = null; e.target.src = 'https://via.placeholder.com/400x400?text=Broken+Link'; }}
                      />
                    </div>
                    <div className="card-content">
                      <div className="product-brand">{product.brand || 'Unknown'}</div>
                      <div className="product-name">{product.product_name || 'No Title'}</div>
                      <div className="card-footer">
                        <div className="product-price">
                          {product.sales_price ? `₹${product.sales_price}` : 'N/A'}
                        </div>
                        {product.score != null && (
                          <div className="product-score">{(product.score).toFixed(4)}</div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </>
  );
}

export default App;
