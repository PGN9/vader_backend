from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from fastapi.responses import JSONResponse
import os
import time
import psutil
import traceback
import platform
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

request_count = 0
start_time = time.time()

app = FastAPI()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Comment(BaseModel):
    id: str
    body: str

class CommentsRequest(BaseModel):
    comments: List[Comment]
    n_clusters: Optional[int] = None  # Allow custom number of clusters

def get_size_in_kb(data):
    return len(data.encode('utf-8')) / 1024  # size in KB

def determine_optimal_clusters(embeddings, max_clusters=10):
    """
    Determine optimal number of clusters using the elbow method
    """
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples - 1, 10)  # Reasonable upper bound
    
    if n_samples <= 2:
        return 1
    elif n_samples <= 5:
        return min(2, n_samples - 1)
    
    # Use elbow method for larger datasets
    inertias = []
    K_range = range(1, max_clusters + 1)
    
    for k in K_range:
        if k >= n_samples:
            break
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
    
    # Simple elbow detection - find the point where improvement diminishes
    if len(inertias) < 3:
        return len(inertias)
    
    # Calculate rate of change
    deltas = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
    delta_deltas = [deltas[i] - deltas[i+1] for i in range(len(deltas)-1)]
    
    # Find elbow point (where second derivative is maximum)
    if delta_deltas:
        elbow_idx = delta_deltas.index(max(delta_deltas)) + 2
        return min(elbow_idx, len(inertias))
    
    return min(3, len(inertias))  # Default fallback

def perform_clustering(embeddings, n_clusters=None):
    """
    Perform K-means clustering on embeddings
    """
    if len(embeddings) <= 1:
        return [0], np.array([[0.0]])
    
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(embeddings)
    else:
        n_clusters = min(n_clusters, len(embeddings))
    
    if n_clusters <= 1:
        return [0] * len(embeddings), np.array([[1.0]])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Calculate silhouette-like scores (distance to cluster center)
    cluster_centers = kmeans.cluster_centers_
    distances = []
    for i, embedding in enumerate(embeddings):
        cluster_id = cluster_labels[i]
        distance = cosine_similarity([embedding], [cluster_centers[cluster_id]])[0][0]
        distances.append(distance)
    
    return cluster_labels.tolist(), distances

def get_cluster_topics(embeddings, texts, cluster_labels, n_clusters):
    """
    Generate topic keywords for each cluster using simple keyword extraction
    """
    cluster_topics = {}
    
    for cluster_id in range(n_clusters):
        # Get texts in this cluster
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_labels[i] == cluster_id]
        
        if not cluster_texts:
            cluster_topics[cluster_id] = []
            continue
        
        # Simple keyword extraction - find most common words
        from collections import Counter
        import re
        
        # Combine all texts in cluster and extract words
        combined_text = ' '.join(cluster_texts).lower()
        words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
                     'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
                     'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 
                     'boy', 'did', 'she', 'use', 'say', 'each', 'which', 'their', 'time', 
                     'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 
                     'well', 'many', 'some', 'what', 'with', 'have', 'this', 'that', 'they',
                     'been', 'said', 'very', 'were', 'more', 'than', 'also', 'back', 'only',
                     'come', 'work', 'life', 'even', 'right', 'down', 'years', 'think', 'where'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Get top keywords
        word_counts = Counter(filtered_words)
        top_keywords = [word for word, count in word_counts.most_common(5)]
        
        cluster_topics[cluster_id] = top_keywords
    
    return cluster_topics

@app.get("/")
def root():
    return {"message": "SentenceTransformer clustering backend is running."}

@app.post("/predict")
def cluster_comments(request: CommentsRequest):
    try:
        # Get process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory_mb = process.memory_info().rss / 1024 / 1024 
        # Track input size
        input_json = json.dumps(request.model_dump()) if hasattr(request, "json") else str(request)
        total_data_size_kb = get_size_in_kb(input_json)

        # Extract text bodies for batch processing
        texts = [comment.body for comment in request.comments]
        
        # Generate embeddings for all texts at once (more efficient)
        embeddings = model.encode(texts)
        
        # Perform clustering
        cluster_labels, similarity_scores = perform_clustering(embeddings, request.n_clusters)
        n_clusters = len(set(cluster_labels))
        
        # Get topic keywords for each cluster
        cluster_topics = get_cluster_topics(embeddings, texts, cluster_labels, n_clusters)
        
        results = []
        for i, comment in enumerate(request.comments):
            cluster_id = cluster_labels[i]
            similarity_score = similarity_scores[i]
            
            results.append({
                "id": comment.id,
                "body": comment.body,
                "cluster": cluster_id,
                "similarity_to_cluster": round(float(similarity_score), 4),
                "cluster_topics": cluster_topics.get(cluster_id, []),
                "embedding_dim": len(embeddings[i])
            })

        # Memory usage check
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        # Peak memory usage (cross-platform)
        if platform.system() == "Windows":
            peak_memory_mb = getattr(process.memory_info(), "peak_wset", current_memory_mb) / 1024 / 1024
        elif platform.system() == "Linux":
            import resource
            peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_kb / 1024
        elif platform.system() == "Darwin":  # macOS
            import resource
            peak_memory_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_memory_mb = peak_memory_bytes / 1024 / 1024
        else:
            peak_memory_mb = current_memory_mb  # fallback

        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_size = cluster_labels.count(cluster_id)
            cluster_stats[cluster_id] = {
                "size": cluster_size,
                "percentage": round((cluster_size / len(results)) * 100, 2),
                "topics": cluster_topics.get(cluster_id, [])
            }

        return_data = {
            "model_used": "sentence-transformers/all-MiniLM-L6-v2",
            "clustering_algorithm": "K-Means",
            "n_clusters": n_clusters,
            "cluster_stats": cluster_stats,
            "results": results,
            "memory_initial_mb": round(initial_memory_mb, 2),
            "memory_peak_mb": round(peak_memory_mb, 2)
        }
        # add data size info
        total_return_size_kb = get_size_in_kb(json.dumps(return_data))
        return_data["total_data_size_kb"] = round(total_data_size_kb, 2)
        return_data["total_return_size_kb"] = round(total_return_size_kb, 2)
        return return_data

    except Exception as e:
        print("Error occurred:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze")  # Add this endpoint for compatibility with your proxy
def analyze_comments(request: CommentsRequest):
    return cluster_comments(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)