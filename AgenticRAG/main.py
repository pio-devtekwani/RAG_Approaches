"""
Agentic RAG POC
================
Stack: LangChain + ChromaDB + OpenAI + PDF
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.live import Live
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

breakpoint()
# Agent imports - try different paths for compatibility
try:
    from langchain.agents import initialize_agent, AgentType
except ImportError:
    initialize_agent = None
    AgentType = None

# ChromaDB imports
import chromadb

console = Console()

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "agentic_rag_docs"


# ─────────────────────────────────────────────
# 2. INGEST PDF INTO CHROMADB
# ─────────────────────────────────────────────
def ingest_pdf(pdf_path: str, azure_config: dict):
    console.print(f"\n[bold cyan]📄 Loading PDF:[/bold cyan] {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    console.print(f"[green]✔ Loaded {len(documents)} pages[/green]")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(documents)
    console.print(f"[green]✔ Split into {len(chunks)} chunks[/green]")

    # Initialize Azure OpenAI embeddings
    embed_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        api_version=azure_config["api_version"]
    )

    with Live(Spinner("dots", text="Creating embeddings & storing in ChromaDB..."), refresh_per_second=10):
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"doc_{i}")
            # Get embedding from Azure OpenAI
            embedding = embed_model.embed_query(chunk.page_content)
            embeddings.append(embedding)
            metadatas.append(chunk.metadata)
            documents_text.append(chunk.page_content)
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

    console.print(f"[green]✔ ChromaDB ready at:[/green] [bold]{CHROMA_DIR}[/bold]\n")
    return client, collection, embed_model


def load_existing_vectorstore(azure_config: dict):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Initialize Azure OpenAI embeddings
    embed_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        api_version=azure_config["api_version"]
    )
    return client, collection, embed_model


# ─────────────────────────────────────────────
# 3. BUILD RETRIEVER TOOL
# ─────────────────────────────────────────────
def build_retriever_tool(collection, embed_model) -> Tool:
    def retrieve_docs(query: str) -> str:
        """Retrieve relevant document chunks from ChromaDB for a given query."""
        # Generate query embedding
        query_embedding = embed_model.embed_query(query)
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4
        )
        
        if not results['documents'] or not results['documents'][0]:
            return "No relevant documents found."
        
        output = []
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "?")
            output.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc}")
        
        return "\n\n---\n\n".join(output)

    return Tool(
        name="retrieve_documents",
        func=retrieve_docs,
        description=(
            "Use this tool to search and retrieve relevant information from the PDF documents. "
            "Input should be a clear search query. Use this multiple times with different queries "
            "if needed to gather complete information."
        ),
    )


# ─────────────────────────────────────────────
# 4. BUILD AGENTIC RAG AGENT
# ─────────────────────────────────────────────
def build_agent(tools: list, azure_config: dict):
    llm = AzureChatOpenAI(
        azure_deployment=azure_config["deployment_name"],
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        api_version=azure_config.get("api_version", "2024-02-15-preview"),
        temperature=0,
    )

    system_message = """You are an intelligent research assistant with access to a document knowledge base.

Your job is to answer user questions using the documents available to you.

INSTRUCTIONS:
1. Always use the `retrieve_documents` tool FIRST to search for relevant information.
2. If the first search doesn't give enough info, search AGAIN with a different or more specific query.
3. Break complex questions into sub-questions and search for each part separately.
4. Combine the retrieved information to form a complete, accurate answer.
5. Always mention which page/source the information came from.
6. If information is not found in the documents, clearly say so.

Be thorough, accurate, and cite your sources."""

    if initialize_agent is not None:
        # Use LangChain agent if available
        agent_executor = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
            agent_kwargs={"prefix": system_message}
        )
        return agent_executor
    else:
        # Fallback: Create a simple wrapper that uses the tool directly
        class SimpleAgent:
            def __init__(self, llm, tool, system_message):
                self.llm = llm
                self.tool = tool
                self.system_message = system_message
            
            def invoke(self, inputs):
                query = inputs["input"]
                chat_history = inputs.get("chat_history", [])
                
                # First, retrieve relevant documents
                docs = self.tool.func(query)
                
                # Then ask LLM to answer based on the docs
                messages = [
                    {"role": "system", "content": self.system_message},
                ]
                
                for msg in chat_history[-4:]:  # Last 2 exchanges
                    if isinstance(msg, HumanMessage):
                        messages.append({"role": "user", "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        messages.append({"role": "assistant", "content": msg.content})
                
                messages.append({
                    "role": "user", 
                    "content": f"Question: {query}\n\nRelevant documents:\n{docs}\n\nPlease answer the question based on the documents above. Cite page numbers."
                })
                
                response = self.llm.invoke(messages)
                return {"output": response.content}
        
        return SimpleAgent(llm, tools[0], system_message)


# ─────────────────────────────────────────────
# 5. CLI CHAT LOOP
# ─────────────────────────────────────────────
def run_chat(agent_executor):
    console.print(Panel.fit(
        "[bold green]🤖 Agentic RAG — CLI Chat[/bold green]\n"
        "[dim]Type your question. Type [bold]exit[/bold] to quit.[/dim]",
        border_style="green"
    ))

    chat_history = []

    while True:
        user_input = Prompt.ask("\n[bold yellow]You[/bold yellow]")
        if user_input.strip().lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        console.print("\n[bold cyan]🔍 Agent is thinking...[/bold cyan]")

        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            answer = result["output"]

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            console.print("\n")
            console.print(Panel(
                Markdown(answer),
                title="[bold green]🤖 Agent Answer[/bold green]",
                border_style="green"
            ))

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# ─────────────────────────────────────────────
# 6. MAIN ENTRY POINT
# ─────────────────────────────────────────────
def main():
    console.print(Panel.fit(
        "[bold magenta]⚡ Agentic RAG POC[/bold magenta]\n"
        "LangChain + ChromaDB + Azure OpenAI + PDF",
        border_style="magenta"
    ))

    # Get Azure OpenAI configuration
    azure_config = {
        "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT") or Prompt.ask("[yellow]Enter Azure OpenAI Endpoint[/yellow]"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY") or Prompt.ask("[yellow]Enter Azure OpenAI API Key[/yellow]", password=True),
        "deployment_name": os.environ.get("AZURE_OPENAI_DEPLOYMENT") or Prompt.ask("[yellow]Enter Azure OpenAI Deployment Name[/yellow]"),
        "api_version": os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    }

    # Check if ChromaDB already has data
    chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())

    if chroma_exists:
        console.print(f"[green]✔ Found existing ChromaDB at {CHROMA_DIR}[/green]")
        choice = Prompt.ask(
            "Load existing data or ingest new PDF?",
            choices=["load", "ingest"],
            default="load"
        )
    else:
        choice = "ingest"

    if choice == "ingest":
        pdf_path = Prompt.ask("[yellow]Enter path to your PDF file[/yellow]")
        if not Path(pdf_path).exists():
            console.print(f"[red]File not found: {pdf_path}[/red]")
            sys.exit(1)
        client, collection, embed_model = ingest_pdf(pdf_path, azure_config)
    else:
        client, collection, embed_model = load_existing_vectorstore(azure_config)
        console.print("[green]✔ Loaded existing vectorstore[/green]")

    # Build tools & agent
    retriever_tool = build_retriever_tool(collection, embed_model)
    agent_executor = build_agent([retriever_tool], azure_config)

    # Start chat
    run_chat(agent_executor)


if __name__ == "__main__":
    main()