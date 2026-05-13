from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import chromadb
from openai import AzureOpenAI
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
load_dotenv(env_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Setup (runs every time) ---
logger.info("Initializing Azure OpenAI client...")
llm_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
logger.info("Azure OpenAI client initialized successfully")

logger.info("Connecting to ChromaDB...")
chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection = chroma_client.get_or_create_collection("my_rag_collection")
logger.info(f"Connected to ChromaDB at {chroma_path} with {collection.count()} existing chunks")


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings using Azure OpenAI."""
    logger.debug(f"Generating embeddings for {len(texts)} text(s)")
    try:
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
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


def index_documents(filepath: str):
    """Load, chunk, embed, and store documents into ChromaDB."""
    logger.info(f"Starting document indexing from: {filepath}")
    
    # Load
    logger.info("Loading documents...")
    loader = PyPDFLoader(filepath)   # Loads PDF files
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} document(s)")

    # Chunk
    logger.info("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Created {len(chunks)} chunks")

    # Embed and store
    logger.info("Generating embeddings using Azure OpenAI...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = get_embeddings(texts)
    logger.info(f"Generated {len(embeddings)} embeddings")

    logger.info("Storing chunks in ChromaDB...")
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(texts))],
        metadatas=[{"source": chunk.metadata.get("source", "")} for chunk in chunks]
    )
    logger.info(f"Successfully stored {collection.count()} chunks in ChromaDB")


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Embed the query and return the top-k most relevant chunks."""
    logger.info(f"Retrieving top {top_k} chunks for query: {query[:100]}...")
    
    logger.debug("Generating query embedding using Azure OpenAI...")
    query_embedding = get_embeddings([query])[0]
    
    logger.debug(f"Querying ChromaDB for {top_k} similar chunks...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    retrieved_chunks = results["documents"][0]
    logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # Log the retrieved chunks for debugging
    for i, chunk in enumerate(retrieved_chunks, 1):
        logger.debug(f"\n--- Chunk {i} ({len(chunk)} chars) ---\n{chunk[:200]}...")
    
    return retrieved_chunks


def rag_answer(query: str) -> str:
    """Retrieve context from ChromaDB and generate an answer via LLM."""
    logger.info(f"Processing RAG query: {query[:100]}...")
    
    logger.info("Retrieving context chunks...")
    context_chunks = retrieve(query)
    context = "\n\n".join(context_chunks)
    logger.info(f"Prepared context from {len(context_chunks)} chunks ({len(context)} characters)")

    logger.debug("Generating LLM prompt...")
    prompt = f"""You are a helpful assistant. Answer the question based on the provided context. 
If the answer is not in the context, say "I don't know." Be thorough and cite the context.

Context:
{context}

Question: {query}
Answer:"""

    logger.info("Sending request to Azure OpenAI...")
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini-imsai",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    logger.info("Received response from LLM")
    return answer


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting RAG Application")
    logger.info("=" * 50)
    
    # Only index if the collection is empty (avoids duplicate ID errors on re-runs)
    if collection.count() == 0:
        logger.info("Collection is empty. Starting document indexing...")
        index_documents(r"D:\projects\TraditionalRAG\Input.pdf")
        logger.info("Document indexing completed")
    else:
        logger.info(f"Collection already has {collection.count()} chunks. Skipping indexing.")

    # Read prompt from input_prompt.txt
    logger.info("Reading user prompt from input_prompt.txt...")
    try:
        with open("input_prompt.txt", "r", encoding="utf-8") as f:
            user_prompt = f.read()
        logger.info(f"Successfully loaded prompt ({len(user_prompt)} characters)")
    except FileNotFoundError:
        logger.error("input_prompt.txt not found")
        raise

    # Process the prompt through RAG
    logger.info("Processing user prompt through RAG...")
    answer = rag_answer(user_prompt)
    print("\n→", answer)
    logger.info("Answer displayed to user")