import requests
import json

# Test data with more diverse comments for better clustering
test_comments = {
    "comments": [
        {
            "id": "test-1",
            "body": "I love this product! It works amazingly well and exceeded my expectations."
        },
        {
            "id": "test-2", 
            "body": "This is terrible, worst experience ever. Complete waste of money."
        },
        {
            "id": "test-3",
            "body": "The delivery was super fast and the packaging was excellent."
        },
        {
            "id": "test-4",
            "body": "Customer service was very helpful and resolved my issue quickly."
        },
        {
            "id": "test-5",
            "body": "Poor quality product, broke after just one week of use."
        },
        {
            "id": "test-6",
            "body": "Great value for money, highly recommend this to others."
        },
        {
            "id": "test-7",
            "body": "The shipping took forever and the item arrived damaged."
        },
        {
            "id": "test-8",
            "body": "Support team was friendly and answered all my questions."
        }
    ],
    "n_clusters": 3  # Optional: specify number of clusters
}

# Send request to your local model backend
try:
    response = requests.post(
        "http://localhost:8000/predict",
        json=test_comments,
        timeout=30
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Clustering backend is working!")
        print(f"Model used: {result.get('model_used', 'Unknown')}")
        print(f"Clustering algorithm: {result.get('clustering_algorithm', 'Unknown')}")
        print(f"Number of clusters: {result.get('n_clusters', 'Unknown')}")
        print(f"Memory usage: {result.get('memory_initial_mb', 'N/A')} MB -> {result.get('memory_peak_mb', 'N/A')} MB")
        
        print("\nCluster Statistics:")
        cluster_stats = result.get('cluster_stats', {})
        for cluster_id, stats in cluster_stats.items():
            print(f"Cluster {cluster_id}: {stats['size']} comments ({stats['percentage']}%)")
            print(f"  Topics: {', '.join(stats['topics'])}")
        
        print("\nDetailed Results:")
        for res in result.get('results', []):
            print(f"ID: {res['id']}")
            print(f"Cluster: {res['cluster']} (similarity: {res['similarity_to_cluster']:.3f})")
            print(f"Topics: {', '.join(res['cluster_topics'])}")
            print(f"Text: {res['body'][:60]}...")
            print("-" * 60)
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to localhost:8000. Make sure your model backend is running.")
except Exception as e:
    print(f"❌ Error: {e}")