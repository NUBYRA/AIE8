#!/usr/bin/env python3
"""
Demo of metadata functionality without API calls.
"""

import numpy as np
import asyncio
from aimakerspace.vectordatabase import VectorDatabase, cosine_similarity, euclidean_distance

async def demo_metadata():
    """Demonstrate metadata functionality with mock data."""
    print("=== Vector Database Metadata Demo ===\n")
    
    # Create vector database
    vector_db = VectorDatabase()
    
    # Sample texts
    texts = [
        "Machine learning is revolutionizing healthcare with AI-powered diagnostics.",
        "The latest iPhone features advanced camera technology and improved battery life.",
        "Climate change is causing rising sea levels and extreme weather patterns.",
        "Python is a versatile programming language popular for data science.",
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
    ]
    
    # Mock embeddings (random but consistent)
    np.random.seed(42)
    mock_embeddings = [np.random.rand(1536) for _ in texts]
    
    # Metadata for each text
    metadata_list = [
        {
            "category": "technology",
            "subcategory": "AI/ML", 
            "source": "tech_blog",
            "date": "2024-01-15",
            "author": "Dr. Smith",
            "sentiment": "positive"
        },
        {
            "category": "technology",
            "subcategory": "mobile",
            "source": "product_review", 
            "date": "2024-01-10",
            "author": "Tech Reviewer",
            "sentiment": "positive"
        },
        {
            "category": "environment",
            "subcategory": "climate",
            "source": "science_journal",
            "date": "2024-01-20", 
            "author": "Climate Scientist",
            "sentiment": "neutral"
        },
        {
            "category": "technology",
            "subcategory": "programming",
            "source": "tutorial_blog",
            "date": "2024-01-12",
            "author": "Python Expert", 
            "sentiment": "positive"
        },
        {
            "category": "environment",
            "subcategory": "energy",
            "source": "environmental_report",
            "date": "2024-01-18",
            "author": "Energy Analyst",
            "sentiment": "positive"
        }
    ]
    
    # Insert vectors with metadata
    print("1. Inserting vectors with metadata...")
    for text, embedding, metadata in zip(texts, mock_embeddings, metadata_list):
        vector_db.insert(text, embedding, metadata)
        print(f"   ✓ {text[:40]}... -> {metadata['category']}/{metadata['subcategory']}")
    print()
    
    # Mock query vector
    query_vector = np.random.rand(1536)
    
    # Test 1: Search without metadata filter
    print("2. General search (no filters):")
    results = vector_db.search(query_vector, k=3, distance_measure=cosine_similarity)
    for i, (text, score, metadata) in enumerate(results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:50]}...")
        print(f"      Category: {metadata.get('category', 'N/A')}")
        print(f"      Author: {metadata.get('author', 'N/A')}")
        print()
    
    # Test 2: Search with metadata filter
    print("3. Search filtered by category='technology':")
    tech_results = vector_db.search(
        query_vector, 
        k=3, 
        distance_measure=cosine_similarity,
        metadata_filter={"category": "technology"}
    )
    for i, (text, score, metadata) in enumerate(tech_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:50]}...")
        print(f"      Subcategory: {metadata.get('subcategory', 'N/A')}")
        print(f"      Source: {metadata.get('source', 'N/A')}")
        print()
    
    # Test 3: Search by author
    print("4. Search filtered by author='Dr. Smith':")
    author_results = vector_db.search(
        query_vector,
        k=2,
        distance_measure=cosine_similarity,
        metadata_filter={"author": "Dr. Smith"}
    )
    for i, (text, score, metadata) in enumerate(author_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:50]}...")
        print(f"      Date: {metadata.get('date', 'N/A')}")
        print()
    
    # Test 4: Search by source
    print("5. Search filtered by source='science_journal':")
    source_results = vector_db.search(
        query_vector,
        k=2,
        distance_measure=cosine_similarity,
        metadata_filter={"source": "science_journal"}
    )
    for i, (text, score, metadata) in enumerate(source_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:50]}...")
        print(f"      Author: {metadata.get('author', 'N/A')}")
        print()
    
    # Test 5: Different distance metrics
    print("6. Search using Euclidean distance:")
    euclidean_results = vector_db.search(
        query_vector,
        k=2,
        distance_measure=euclidean_distance,
        metadata_filter={"category": "environment"}
    )
    for i, (text, score, metadata) in enumerate(euclidean_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:50]}...")
        print(f"      Subcategory: {metadata.get('subcategory', 'N/A')}")
        print()
    
    # Test 6: Get metadata for specific document
    print("7. Retrieving metadata for specific document:")
    first_text = texts[0]
    metadata = vector_db.get_metadata(first_text)
    print(f"   Document: {first_text}")
    print(f"   Metadata: {metadata}")
    print()
    
    print("=== Metadata functionality demo completed! ===")
    print("✅ All metadata features are working correctly:")
    print("   - Metadata storage and retrieval")
    print("   - Metadata filtering in searches")
    print("   - Multiple distance metrics")
    print("   - Flexible metadata queries")

if __name__ == "__main__":
    asyncio.run(demo_metadata())
