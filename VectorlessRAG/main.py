from pageindex import PageIndexClient
import pageindex.utils as utils
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
import json
import logging
from datetime import datetime

# Load environment variables from root .env file
root_dir = Path(__file__).parent.parent
env_file = root_dir / ".env"
load_dotenv(env_file)

# Ensure logs directory exists BEFORE configuring logging
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/vectorless_rag_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info(f"Environment variables loaded from {env_file}")

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY")
if not PAGEINDEX_API_KEY:
    logger.error("PAGEINDEX_API_KEY environment variable not found")
    raise ValueError("PAGEINDEX_API_KEY is required")
logger.info("PageIndex API key loaded")

pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
logger.info("PageIndex client initialized")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not all([azure_openai_api_key, azure_openai_api_version, azure_openai_endpoint]):
    logger.error("Azure OpenAI environment variables not found")
    raise ValueError("Azure OpenAI credentials are required")
logger.info("Azure OpenAI credentials loaded")

llm_client = AzureOpenAI(
    api_key=azure_openai_api_key,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint
)
logger.info("Azure OpenAI client initialized")


def submit_pdf_document(pdf_path):
    """
    Submit a PDF document for generating PageIndex tree.
    
    Args:
        pdf_path (str): Path to the PDF file to submit
        
    Returns:
        str: The document ID returned by PageIndex
        
    Raises:
        FileNotFoundError: If the PDF file does not exist
        Exception: If submission fails
    """
    try:
        logger.info(f"Starting document submission: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        logger.debug(f"Submitting document to PageIndex: {pdf_path}")
        doc_id = pi_client.submit_document(pdf_path)["doc_id"]
        
        
        logger.info(f"Document successfully submitted. Document ID: {doc_id}")
        return doc_id
        
    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error submitting document: {str(e)}", exc_info=True)
        raise

def retrieve_document_tree(doc_id):
    """
    Retrieve the tree structure of a document using its document ID.
    
    Args:
        doc_id (str): The document ID to retrieve the tree for
        
    Returns:
        dict: The tree structure if ready, None if document is still processing
    """
    try:
        logger.info(f"Checking retrieval status for document ID: {doc_id}")
        
        if pi_client.is_retrieval_ready(doc_id):
            logger.info(f"Document {doc_id} is ready for retrieval")
            logger.debug(f"Retrieving tree structure with node summaries...")
            
            tree_response = pi_client.get_tree(doc_id, node_summary=True)['result']
            tree = tree_response['result']
            
            logger.info(f"Successfully retrieved tree structure")
            logger.debug(f"Printing tree structure...")
            utils.print_tree(tree)
            
            return tree
        else:
            logger.warning(f"Document {doc_id} is still processing. Please try again later")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving document tree for {doc_id}: {str(e)}", exc_info=True)
        return None

def call_llm(prompt):
    """
    Call Azure OpenAI LLM with the given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The response from the LLM
        
    Raises:
        Exception: If LLM call fails
    """
    try:
        logger.info("Calling Azure OpenAI LLM")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini-imsai",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        logger.info(f"LLM call successful. Response length: {len(result)} characters")
        logger.debug(f"LLM response: {result[:200]}...")  # Log first 200 chars
        
        return result
        
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
        raise

def query_document(query, tree):
    """
    Query the document tree to find relevant nodes for the given question.
    
    Args:
        query (str): The question to search for in the document
        tree (dict): The tree structure of the document
        
    Returns:
        dict: A dictionary containing thinking process and list of relevant node IDs
    """
    try:
        logger.info(f"Starting document query: '{query}'")
        
        # Remove text fields from tree for the LLM prompt
        logger.debug("Removing text fields from tree...")
        tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])
        logger.info("Tree processed - text fields removed")
        
        # Create the search prompt for the LLM
        search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find all nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2)}

Give respose accorrding to the Question only.
"""
        
        logger.debug(f"Search prompt created. Size: {len(search_prompt)} characters")
        
        # Call the LLM to get relevant nodes
        logger.info("Sending query to LLM...")
        tree_search_result = call_llm(search_prompt)
        
        # Parse the JSON response
        logger.info("Parsing LLM response...")
        result = json.loads(tree_search_result)
        
        relevant_nodes = result.get('node_list', [])
        logger.info(f"Query completed successfully")
        logger.info(f"  Query: {query}")
        logger.info(f"  Found {len(relevant_nodes)} relevant nodes: {relevant_nodes}")
        logger.debug(f"  LLM Thinking: {result.get('thinking', 'N/A')}")
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {str(e)}", exc_info=True)
        logger.error(f"Raw response: {tree_search_result[:500]}...")
        return {"thinking": "Error - Invalid JSON response", "node_list": []}
    except Exception as e:
        logger.error(f"Error during document query: {str(e)}", exc_info=True)
        return {"thinking": "Error - Query failed", "node_list": []}

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Vectorless RAG Application")
    logger.info("=" * 60)
    
    try:
        # Step 1: Use local PDF file
        logger.info("\n[STEP 1] Loading local PDF document...")
        pdf_path = r"D:\projects\VectorlessRAG\Input.pdf"
        
        logger.info(f"PDF path: {pdf_path}")
        
        # Verify file exists
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found at: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        file_size = os.path.getsize(pdf_path)
        logger.info(f"PDF file size: {file_size} bytes")
        
        # Submit document
        # logger.info("\n[STEP 2] Submitting document for processing...")
        # doc_id = submit_pdf_document(pdf_path)
        doc_id = "pi-cmp3w3ln6001801qwycz1mv7g"
        
        # Step 3: Retrieve document tree
        logger.info("\n[STEP 3] Retrieving document tree structure...")
        tree = retrieve_document_tree(doc_id)
        
        if tree is None:
            logger.warning("Document tree not ready. Please try again later.")
        else:
            # Step 4: Query the document
            logger.info("\n[STEP 4] Reading query from file...")
            query_file_path = "query.txt"
            
            if not os.path.exists(query_file_path):
                logger.error(f"Query file not found: {query_file_path}")
                raise FileNotFoundError(f"Query file not found: {query_file_path}")
            
            with open(query_file_path, 'r', encoding='utf-8') as f:
                query = f.read().strip()
            
            logger.info(f"Query loaded successfully ({len(query)} characters)")
            logger.debug(f"Query preview: {query[:200]}...")
            
            logger.info("\n[STEP 5] Querying document for relevant information...")
            logger.info(f"Processing query...")
            
            query_result = query_document(query, tree)
            
            logger.info("\n" + "=" * 60)
            logger.info("Application completed successfully")
            logger.info("=" * 60)
            
    except FileNotFoundError as e:
        logger.error(f"File not found error: {str(e)}", exc_info=True)
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        raise
