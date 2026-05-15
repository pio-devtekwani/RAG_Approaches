"""
Product Recommendation System using LLM + RAG
==============================================
A smart product recommendation engine that uses RAG to:
- Find products based on user queries
- Recommend similar products
- Provide personalized suggestions with explanations
- Support natural language product searches

Stack: ChromaDB + Azure OpenAI + LangChain
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

import chromadb
from openai import AzureOpenAI

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
load_dotenv(env_file)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/product_recommendation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Azure OpenAI client
logger.info("Initializing Azure OpenAI client...")
llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
logger.info("Azure OpenAI client initialized successfully")

# Initialize ChromaDB
logger.info("Connecting to ChromaDB...")
chroma_path = "./chroma_db"
os.makedirs(chroma_path, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection = chroma_client.get_or_create_collection(
    name="product_recommendations",
    metadata={"hnsw:space": "cosine"}
)
logger.info(f"Connected to ChromaDB at {chroma_path}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Azure OpenAI."""
    logger.debug(f"Generating embeddings for {len(texts)} text(s)")
    try:
        embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        response = llm_client.embeddings.create(
            model=embedding_model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Successfully generated {len(embeddings)} embedding(s)")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


def create_product_document(product: Dict[str, Any]) -> str:
    """
    Create a rich text representation of a product for embedding.
    This helps the LLM understand product context better.
    """
    doc = f"""Product: {product['name']}
Category: {product['category']}
Price: ${product['price']}
Brand: {product.get('brand', 'N/A')}
Rating: {product.get('rating', 'N/A')}/5.0
Description: {product['description']}
Features: {', '.join(product.get('features', []))}
Tags: {', '.join(product.get('tags', []))}
"""
    if 'specifications' in product:
        specs = ', '.join([f"{k}: {v}" for k, v in product['specifications'].items()])
        doc += f"Specifications: {specs}\n"
    
    return doc.strip()


def index_products(products_file: str = "sample_products.json"):
    """Load products from JSON file and index them into ChromaDB."""
    logger.info(f"Starting product indexing from: {products_file}")
    
    # Load products
    products_path = Path(__file__).parent / products_file
    if not products_path.exists():
        logger.error(f"Products file not found: {products_path}")
        raise FileNotFoundError(f"Products file not found: {products_path}")
    
    with open(products_path, 'r', encoding='utf-8') as f:
        products = json.load(f)
    
    logger.info(f"Loaded {len(products)} products from {products_file}")
    
    # Create product documents
    logger.info("Creating product documents for embedding...")
    product_docs = []
    product_ids = []
    product_metadatas = []
    
    for product in products:
        doc = create_product_document(product)
        product_docs.append(doc)
        product_ids.append(f"product_{product['id']}")
        product_metadatas.append({
            "id": str(product['id']),
            "name": product['name'],
            "category": product['category'],
            "price": str(product['price']),
            "rating": str(product.get('rating', 'N/A')),
            "brand": product.get('brand', 'N/A')
        })
    
    # Generate embeddings
    logger.info("Generating embeddings using Azure OpenAI...")
    embeddings = get_embeddings(product_docs)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Store in ChromaDB
    logger.info("Storing products in ChromaDB...")
    collection.add(
        documents=product_docs,
        embeddings=embeddings,
        ids=product_ids,
        metadatas=product_metadatas
    )
    logger.info(f"Successfully indexed {collection.count()} products in ChromaDB")


def retrieve_products(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top-k most relevant products for a given query."""
    logger.info(f"Retrieving top {top_k} products for query: {query[:100]}...")
    
    # Generate query embedding
    logger.debug("Generating query embedding using Azure OpenAI...")
    query_embedding = get_embeddings([query])[0]
    
    # Query ChromaDB
    logger.debug(f"Querying ChromaDB for {top_k} similar products...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Format results
    products = []
    for i in range(len(results['documents'][0])):
        products.append({
            'document': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i] if 'distances' in results else None
        })
    
    logger.info(f"Retrieved {len(products)} relevant products")
    return products


def generate_recommendation(query: str, products: List[Dict[str, Any]]) -> str:
    """Generate personalized product recommendations using LLM."""
    logger.info(f"Generating recommendations for: {query[:100]}...")
    
    # Build context from retrieved products
    context = "\n\n".join([
        f"Product {i+1}:\n{prod['document']}"
        for i, prod in enumerate(products)
    ])
    
    # Create prompt
    prompt = f"""You are an expert product recommendation assistant. Based on the user's query and the product catalog, provide personalized recommendations with detailed explanations.

User Query: {query}

Available Products:
{context}

Instructions:
1. Recommend the most suitable products from the list above
2. Explain WHY each product matches the user's needs
3. Compare key features, prices, and ratings
4. Provide a clear ranking of recommendations
5. Be specific and helpful
6. If the user's query mentions preferences (budget, features, brand), prioritize accordingly

Provide your recommendations:"""

    logger.info("Sending request to Azure OpenAI for recommendation generation...")
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-docgen"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful product recommendation expert who provides personalized, well-reasoned product suggestions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        
        recommendation = response.choices[0].message.content
        logger.info("Successfully generated recommendations")
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise


def recommend_products(query: str, top_k: int = 5) -> str:
    """Main function to get product recommendations based on user query."""
    logger.info("=" * 60)
    logger.info(f"Processing recommendation request: {query}")
    logger.info("=" * 60)
    
    # Retrieve relevant products
    products = retrieve_products(query, top_k)
    
    # Generate recommendations using LLM
    recommendation = generate_recommendation(query, products)
    
    return recommendation


def display_recommendations(query: str, recommendation: str):
    """Display recommendations in a formatted way."""
    print("\n" + "=" * 80)
    print(f"🔍 QUERY: {query}")
    print("=" * 80)
    print("\n💡 RECOMMENDATIONS:\n")
    print(recommendation)
    print("\n" + "=" * 80)


def interactive_mode():
    """Run the recommendation system in interactive mode."""
    logger.info("Starting interactive product recommendation mode...")
    print("\n" + "=" * 80)
    print("🛍️  PRODUCT RECOMMENDATION SYSTEM")
    print("=" * 80)
    print("\nWelcome! I can help you find the perfect products.")
    print("Examples:")
    print("  - 'I need a laptop for gaming under $1500'")
    print("  - 'Recommend wireless headphones with good battery life'")
    print("  - 'Find me a smartphone with best camera'")
    print("\nType 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            query = input("👤 Your request: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Product Recommendation System! 👋\n")
                break
            
            if not query:
                continue
            
            # Get recommendations
            recommendation = recommend_products(query)
            display_recommendations(query, recommendation)
            
        except KeyboardInterrupt:
            print("\n\nExiting... 👋\n")
            break
        except Exception as e:
            logger.error(f"Error in interactive mode: {e}")
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("PRODUCT RECOMMENDATION SYSTEM - STARTING")
    logger.info("=" * 60)
    
    try:
        # Check if products are already indexed
        if collection.count() == 0:
            logger.info("Collection is empty. Starting product indexing...")
            index_products("sample_products.json")
            logger.info("Product indexing completed successfully")
        else:
            logger.info(f"Collection already has {collection.count()} products. Skipping indexing.")
        
        # Check if query.txt exists for batch mode
        query_file = Path(__file__).parent / "query.txt"
        if query_file.exists():
            logger.info("Found query.txt - Running in batch mode")
            with open(query_file, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            
            if query:
                logger.info(f"Processing query from file: {query[:100]}...")
                recommendation = recommend_products(query)
                display_recommendations(query, recommendation)
            else:
                logger.warning("query.txt is empty. Starting interactive mode...")
                interactive_mode()
        else:
            logger.info("No query.txt found. Starting interactive mode...")
            interactive_mode()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    
    logger.info("=" * 60)
    logger.info("PRODUCT RECOMMENDATION SYSTEM - COMPLETED")
    logger.info("=" * 60)
