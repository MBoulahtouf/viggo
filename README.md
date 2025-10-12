# Viggo: The Spoiler-Free Lore Companion

Viggo is an intelligent, context-aware Q&A tool designed to help you explore the lore of books, games, and other documents without spoiling the plot. Ask a question, tell Viggo where you are in the story, and get safe, spoiler-free answers.

## About The Project

This project was built to solve a common problem for fans of epic stories: you want to understand a specific detail or character, but searching online almost always leads to spoilers.

Viggo uses a Retrieval-Augmented Generation (RAG) architecture to ensure its answers are based *only* on the content you have already read.

**Core Features:**
* Upload a document (PDF) to serve as the knowledge base.
* Ask questions in natural language.
* Provide your current page number to activate the **spoiler guardrail**.
* Receive answers generated from the text up to your current location.

### Tech Stack

* **Backend:** FastAPI
* **Dependency Management:** Poetry + uv (hybrid approach for speed and reproducibility)
* **Core AI/ML Libraries:** LangChain, Sentence-Transformers (for embeddings)
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **PDF Processing:** pypdf

---

## Getting Started

Follow these instructions to get a local copy up and running for development and testing.

### Prerequisites

* Python 3.12
* [Poetry](https://python-poetry.org/docs/#installation) or [uv](https://docs.astral.sh/uv/getting-started/installation/) installed on your system.

**Note:** For faster dependency installation, we recommend using `uv`. The CI uses `uv` for ultra-fast builds.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone MBoulahtouf/viggo
    cd viggo
    ```
2.  **Install dependencies:**
    
    **Option A: Using uv (Recommended - Faster)**
    ```sh
    uv sync --all-extras
    ```
    
    **Option B: Using Poetry**
    ```sh
    poetry install
    ```

### Running the Application

**Using uv:**
```sh
uv run uvicorn viggo.main:app --reload
```

**Using Poetry:**
```sh
poetry run uvicorn viggo.main:app --reload
```

## Quick Start

For a quick setup, see our [Quick Start Guide](docs/QUICKSTART.md).

## Documentation

- [Development Guide](docs/DEVELOPMENT.md) - Comprehensive development setup and guidelines
- [API Documentation](http://127.0.0.1:8000/docs) - Interactive API docs (when server is running)

## Environment Setup

1. Copy the environment template:
   ```bash
   cp .env.sample .env
   ```

2. Edit `.env` with your configuration:
   ```bash
   # Required
   GROQ_API_KEY=your_groq_api_key_here
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password_here
   
   # Optional
   MLFLOW_TRACKING_URI=http://localhost:5000
   WANDB_PROJECT=viggo-lore-companion
   DATA_DIR=/absolute/path/to/data
   ```

3. Download the spaCy model:
   
   **Using uv:**
   ```bash
   uv run python -m spacy download en_core_web_sm
   ```
   
   **Using Poetry:**
   ```bash
   poetry run python -m spacy download en_core_web_sm
   ```

## Multi-Agent Framework

Viggo now features a sophisticated multi-agent framework that enhances query processing and response generation:

- **Query Analyzer Agent**: Intelligent intent detection (character, plot, setting, relationship queries)
- **Entity Extractor Agent**: Advanced entity and relationship extraction from text
- **Context Aggregator Agent**: Hybrid context aggregation from multiple sources
- **Response Generator Agent**: Template-based response generation with fallback mechanisms
- **Azure GraphRAG Service**: Advanced relationship extraction using Microsoft GraphRAG approach

The framework follows SOLID principles and integrates seamlessly with Azure Search and Neo4j for enhanced retrieval capabilities.

## Testing

Run the comprehensive test suite:

**Using uv:**
```bash
# Run all tests
uv run pytest

# Run multi-agent specific tests
uv run pytest tests/test_multi_agent_*.py -v

# Run with coverage
uv run pytest --cov=viggo --cov-report=html
```

**Using Poetry:**
```bash
# Run all tests
poetry run pytest

# Run multi-agent specific tests
poetry run pytest tests/test_multi_agent_*.py -v

# Run with coverage
poetry run pytest --cov=viggo --cov-report=html
```

### Test Categories
- **Core Tests**: Basic functionality and SOLID principles compliance
- **Framework Tests**: Full multi-agent integration testing
- **Integration Tests**: End-to-end system testing
- **Unit Tests**: Individual component testing

For more testing information, see the [Test Documentation](tests/README.md) and [Development Guide](docs/DEVELOPMENT.md#testing).

## CI/CD Pipeline

Viggo includes a comprehensive CI/CD pipeline with GitHub Actions:
- **Multi-Agent Tests**: Dedicated testing for the multi-agent framework
- **Quality Checks**: Linting (Ruff), type checking (MyPy), security scanning (Bandit)
- **Code Coverage**: Automated coverage reporting with Codecov
- **Dependency Management**: Ultra-fast uv-based dependency management with Poetry lock file compatibility
- **Performance**: CI runs 2-3x faster with uv compared to traditional Poetry setup

## Architecture

Viggo uses a hybrid RAG architecture combining:
- **Vector Search**: Azure Search for semantic similarity
- **Graph Search**: Neo4j for relationship-based retrieval
- **Multi-Agent Processing**: Specialized agents for different query types
- **Spoiler Protection**: Page-based content filtering
