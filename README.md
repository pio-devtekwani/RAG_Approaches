# RAG Approaches

A comprehensive exploration of Retrieval-Augmented Generation (RAG) implementations using different approaches and architectures.

## 📋 Project Overview

This repository contains three distinct RAG implementations:

- **GraphRAG**: Advanced graph-based RAG using knowledge graphs for contextual retrieval
- **TraditionalRAG**: Classical vector-based RAG with semantic search
- **VectorlessRAG**: Keyword and metadata-based RAG without vector embeddings

Each approach has unique strengths and is optimized for different use cases.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- API keys (OpenAI or Azure OpenAI)

### Installation

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd RAG_Approaches
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

## 📁 Project Structure

```
RAG_Approaches/
├── GraphRag_Poc/           # Graph-based RAG implementation
│   ├── pdf_2_text.py       # PDF to text conversion
│   ├── settings.yaml       # GraphRAG settings
│   ├── cache/              # Cached computations
│   ├── input/              # Input documents
│   ├── output/             # Generated graphs and embeddings
│   ├── prompts/            # System prompts for graph extraction
│   └── logs/               # Execution logs
├── TraditionalRAG/         # Vector-based RAG implementation
│   ├── main.py             # Main entry point
│   ├── chroma_db/          # Chroma vector database
│   └── test.txt            # Test data
├── VectorlessRAG/          # Keyword-based RAG implementation
│   ├── main.py             # Main entry point
│   ├── notebook.ipynb      # Jupyter notebook
│   └── logs/               # Execution logs
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not committed)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Configuration

Create a `.env` file in the root directory with the following variables:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here

# Model Configuration
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-3-small

# Database Paths
CHROMA_DB_PATH=./TraditionalRAG/chroma_db
LANCEDB_PATH=./GraphRag_Poc/output/lancedb

# Other Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
```

## 📚 Usage

### GraphRAG Approach

```bash
cd GraphRag_Poc
python main.py --input ./input/input.txt --mode full
```

**Best for:**
- Complex multi-hop queries
- Knowledge graph visualization
- Community detection in documents

### Traditional RAG Approach

```bash
cd TraditionalRAG
python main.py --query "Your question here"
```

**Best for:**
- Fast semantic searches
- Dense document retrieval
- Well-structured documents

### Vectorless RAG Approach

```bash
cd VectorlessRAG
python main.py --query "Your question here"
```

**Best for:**
- Keyword-based searches
- Low computational overhead
- Structured data retrieval

## 📊 Comparison

| Approach | Speed | Accuracy | Memory | Complexity |
|----------|-------|----------|--------|-----------|
| GraphRAG | Medium | High | High | High |
| Traditional RAG | Fast | Medium-High | Medium | Medium |
| Vectorless RAG | Very Fast | Medium | Low | Low |

## 🛠️ Key Features

- **Multiple RAG Implementations**: Compare different retrieval strategies
- **Caching System**: Efficient computation caching to reduce API calls
- **Configurable**: Easy to customize models, prompts, and parameters
- **Logging**: Comprehensive logging for debugging and analysis
- **Modular Design**: Reusable components across implementations

## 📦 Dependencies

Core dependencies:
- `langchain` - LLM framework
- `openai` or `azure-openai` - LLM API clients
- `chromadb` - Vector database
- `lancedb` - Vector database alternative
- `python-dotenv` - Environment variable management

See `requirements.txt` for complete list.

## 🧪 Testing

Run tests (if available):
```bash
pytest tests/
```

## 📝 Documentation

- GraphRAG configuration: See `GraphRag_Poc/settings.yaml`
- Custom prompts: See `GraphRag_Poc/prompts/` directory
- Input format: See `GraphRag_Poc/input/` for examples

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤔 FAQ

**Q: Which approach should I use?**
A: Choose based on your needs:
- Need graph insights? → GraphRAG
- Need speed and accuracy? → Traditional RAG
- Need minimal resources? → Vectorless RAG

**Q: How do I add new documents?**
A: Place files in the respective `input/` directory and run the main scripts.

**Q: Can I combine approaches?**
A: Yes! Create a hybrid implementation by integrating multiple retrieval methods.

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Last Updated:** May 2026
