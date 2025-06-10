# Strider: The Spoiler-Free Lore Companion

Strider is an intelligent, context-aware Q&A tool designed to help you explore the lore of books, games, and other documents without spoiling the plot. Ask a question, tell Strider where you are in the story, and get safe, spoiler-free answers.

## About The Project

This project was built to solve a common problem for fans of epic stories: you want to understand a specific detail or character, but searching online almost always leads to spoilers.

Strider uses a Retrieval-Augmented Generation (RAG) architecture to ensure its answers are based *only* on the content you have already read.

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
    git clone MBoulahtouf/strider
    cd strider
    ```
2.  **Install dependencies:**
    Poetry will create a virtual environment and install all necessary packages from the `pyproject.toml` file.
    ```sh
    poetry install
    ```

### Running the Application

To run the FastAPI server, use the `poetry run` command:
```sh
poetry run uvicorn strider.main:app --reload
