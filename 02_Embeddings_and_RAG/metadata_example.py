#!/usr/bin/env python3
"""
Example demonstrating metadata support in the enhanced VectorDatabase.
"""

import asyncio
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.embedding import EmbeddingModel

async def main():
    # Sample documents with metadata
    documents = [
        "Machine learning is revolutionizing healthcare with AI-powered diagnostics.",
        "The latest iPhone features advanced camera technology and improved battery life.",
        "Climate change is causing rising sea levels and extreme weather patterns.",
        "Python is a versatile programming language popular for data science.",
        "Renewable energy sources like solar and wind are becoming more cost-effective.",
    ]
    
    # Metadata for each document
    metadata_list = [
        {
            "category": "technology",
            "subcategory": "AI/ML",
            "source": "tech_blog",
            "date": "2024-01-15",
            "author": "Dr. Smith"
        },
        {
            "category": "technology", 
            "subcategory": "mobile",
            "source": "product_review",
            "date": "2024-01-10",
            "author": "Tech Reviewer"
        },
        {
            "category": "environment",
            "subcategory": "climate",
            "source": "science_journal",
            "date": "2024-01-20",
            "author": "Climate Scientist"
        },
        {
            "category": "technology",
            "subcategory": "programming",
            "source": "tutorial_blog",
            "date": "2024-01-12",
            "author": "Python Expert"
        },
        {
            "category": "environment",
            "subcategory": "energy",
            "source": "environmental_report",
            "date": "2024-01-18",
            "author": "Energy Analyst"
        }
    ]
    
    # Create vector database with metadata
    vector_db = VectorDatabase()
    await vector_db.abuild_from_list(documents, metadata_list)
    
    print("=== Vector Database with Metadata Support ===\n")
    
    # Example 1: Search without filters
    print("1. General search for 'technology':")
    results = vector_db.search_by_text("technology", k=3)
    for i, (text, score, metadata) in enumerate(results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:60]}...")
        print(f"      Category: {metadata.get('category', 'N/A')}")
        print(f"      Source: {metadata.get('source', 'N/A')}")
        print()
    
    # Example 2: Search with metadata filter
    print("2. Search for 'technology' but only in tech blogs:")
    tech_blog_results = vector_db.search_by_text(
        "technology", 
        k=3, 
        metadata_filter={"source": "tech_blog"}
    )
    for i, (text, score, metadata) in enumerate(tech_blog_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:60]}...")
        print(f"      Author: {metadata.get('author', 'N/A')}")
        print()
    
    # Example 3: Search by category
    print("3. Search for 'environment' content:")
    env_results = vector_db.search_by_text(
        "environment", 
        k=2, 
        metadata_filter={"category": "environment"}
    )
    for i, (text, score, metadata) in enumerate(env_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text[:60]}...")
        print(f"      Subcategory: {metadata.get('subcategory', 'N/A')}")
        print()
    
    # Example 4: Get metadata for specific document
    print("4. Metadata for a specific document:")
    specific_metadata = vector_db.get_metadata(documents[0])
    print(f"   Document: {documents[0]}")
    print(f"   Metadata: {specific_metadata}")
    print()
    
    # Example 5: Search by author
    print("5. Search for content by 'Climate Scientist':")
    author_results = vector_db.search_by_text(
        "climate", 
        k=2, 
        metadata_filter={"author": "Climate Scientist"}
    )
    for i, (text, score, metadata) in enumerate(author_results, 1):
        print(f"   {i}. Score: {score:.3f}")
        print(f"      Text: {text}")
        print(f"      Date: {metadata.get('date', 'N/A')}")
        print()

if __name__ == "__main__":
    asyncio.run(main())
