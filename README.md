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
* **Dependency Management:** Poetry
* **Core AI/ML Libraries:** LangChain, Sentence-Transformers (for embeddings)
* **Vector Search:** FAISS (Facebook AI Similarity Search)
* **PDF Processing:** pypdf

---

## Getting Started

Follow these instructions to get a local copy up and running for development and testing.

### Prerequisites

* Python 3.12
* [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone MBoulahtouf/viggo
    cd viggo
    ```
2.  **Install dependencies:**
    Poetry will create a virtual environment and install all necessary packages from the `pyproject.toml` file.
    ```sh
    poetry install
    ```

### Running the Application

To run the FastAPI server, use the `poetry run` command:
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
   ```bash
   poetry run python -m spacy download en_core_web_sm
   ```

## Testing

Run the test suite:
```bash
poetry run pytest
```

For more testing information, see the [Development Guide](docs/DEVELOPMENT.md#testing).
