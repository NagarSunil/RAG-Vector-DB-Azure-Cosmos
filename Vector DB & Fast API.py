'''Overview
Set up Azure OpenAI and Vector Database: Use Azure OpenAI for embeddings and Azure Cosmos DB (or alternatives like Pinecone or Qdrant) to store vectors.
Generate and Store Embeddings: Embed your documents using the Azure OpenAI embedding model and save them in the vector database.
Query for Similarity: Retrieve relevant documents based on vector similarity.
Integrate FastAPI: Build a RESTful API using FastAPI to expose endpoints for storing and querying data.
Combine with Azure OpenAI GPT Model: Use the retrieved documents as context to answer user queries using Azure OpenAI's GPT model.'''


## 1: Set up Environment
AZURE_OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment-name
AZURE_COSMOS_URL=https://your-cosmosdb.documents.azure.com:443/
AZURE_COSMOS_KEY=your-cosmosdb-primary-key
DATABASE_NAME=vector_db
CONTAINER_NAME=embeddings

## 1. Initialize Azure OpenAI and Cosmos DB:
import os
from dotenv import load_dotenv
import openai
from azure.cosmos import CosmosClient

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2024-05-01-preview"

# Cosmos DB Configuration
cosmos_url = os.getenv("AZURE_COSMOS_URL")
cosmos_key = os.getenv("AZURE_COSMOS_KEY")
database_name = os.getenv("DATABASE_NAME")
container_name = os.getenv("CONTAINER_NAME")

# Initialize Cosmos DB client
cosmos_client = CosmosClient(cosmos_url, cosmos_key)
database = cosmos_client.create_database_if_not_exists(id=database_name)
container = database.create_container_if_not_exists(id=container_name, partition_key="/id")


## 2. Generate and Store Embeddings:
def generate_embeddings(texts):
    """
    Generate embeddings using Azure OpenAI's embedding model.
    """
    response = openai.Embedding.create(
        input=texts,
        engine=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    )
    return [data['embedding'] for data in response['data']]

def store_embeddings(texts):
    """
    Store documents and embeddings in Cosmos DB.
    """
    embeddings = generate_embeddings(texts)
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        container.upsert_item({
            "id": str(i),
            "text": text,
            "embedding": embedding
        })
    print("Embeddings stored successfully.")

# Example texts
documents = [
    "Azure OpenAI provides GPT models for text generation.",
    "Cosmos DB is a globally distributed NoSQL database.",
    "Retrieval-Augmented Generation enhances AI applications."
]

# Store embeddings
store_embeddings(documents)


## 3 Query for Similar Documents

import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def query_similar_documents(query, top_k=2):
    """
    Query for the most similar documents based on embeddings.
    """
    # Generate embedding for the query
    query_embedding = generate_embeddings([query])[0]
    
    # Retrieve all items from Cosmos DB
    query_results = container.query_items("SELECT * FROM embeddings", enable_cross_partition_query=True)
    
    # Calculate similarity
    similarities = []
    for item in query_results:
        stored_embedding = np.array(item['embedding'])
        similarity = cosine_similarity(query_embedding, stored_embedding)
        similarities.append((item['text'], similarity))
    
    # Sort and return top-k documents
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Query example
query_text = "How does RAG improve AI responses?"
results = query_similar_documents(query_text)
print("Top relevant documents:", results)

## 4 Build a FastAPI Service

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request body schemas
class StoreRequest(BaseModel):
    texts: list

class QueryRequest(BaseModel):
    query: str

@app.post("/store")
def store_texts(request: StoreRequest):
    """
    Endpoint to store text embeddings.
    """
    store_embeddings(request.texts)
    return {"message": "Documents stored successfully."}

@app.post("/query")
def query_text(request: QueryRequest):
    """
    Endpoint to query similar documents and return GPT response.
    """
    # Retrieve top relevant documents
    top_docs = query_similar_documents(request.query)
    context = "\n".join([doc[0] for doc in top_docs])
    
    # Generate response using GPT model
    prompt = f"Context:\n{context}\n\nQuestion: {request.query}\nAnswer:"
    response = openai.ChatCompletion.create(
        engine="gpt-4",  # Your GPT deployment
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return {"response": response['choices'][0]['message']['content']}

# Run the FastAPI app
# Command: uvicorn app:app --reload

## 5: Test the Application

## Run the FastAPI 
uvicorn app:app --reload

## Test Endpoints:
    ## Store Documents:
POST http://127.0.0.1:8000/store
Content-Type: application/json

{
  "texts": ["Azure OpenAI generates embeddings", "Cosmos DB stores vectors efficiently"]
}

## Query Documents:
POST http://127.0.0.1:8000/query
Content-Type: application/json

{
  "query": "How does Cosmos DB store embeddings?"
}


############################################################################################################################################
''' Summary of Workflow
Azure OpenAI generates embeddings for text.
Cosmos DB stores these embeddings along with the text.
FastAPI provides RESTful endpoints to store new documents, query relevant documents, and generate GPT-based responses.
RAG Pipeline combines document retrieval and GPT to produce accurate, context-aware answers.
This approach ensures efficient and scalable implementation for RAG applications. '''

