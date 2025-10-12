# Quick Start Guide

Get Viggo up and running in 5 minutes!

## Prerequisites

- Python 3.11 or 3.12
- Poetry or uv (uv recommended for faster setup)
- Docker (for Neo4j)

## 1. Clone and Install

```bash
git clone <repository-url>
cd viggo
```

**Option A: Using uv (Recommended - Faster)**
```bash
uv sync --all-extras
uv run python -m spacy download en_core_web_sm
```

**Option B: Using Poetry**
```bash
poetry install
poetry run python -m spacy download en_core_web_sm
```

## 2. Start Neo4j

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

## 3. Configure Environment

```bash
cp .env.sample .env
```

Edit `.env`:
```bash
GROQ_API_KEY=your_groq_api_key_here
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

## 4. Start the Server

**Using uv:**
```bash
uv run uvicorn viggo.main:app --reload
```

**Using Poetry:**
```bash
poetry run uvicorn viggo.main:app --reload
```

## 5. Test the API

**Upload a document:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

**Query the document:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main character?", "page_number": 50}'
```

**Explore the knowledge graph:**
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/graph/nodes?label=Character"
```

## 6. View API Documentation

Open your browser to:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Troubleshooting

**Neo4j connection issues:**
- Check if Neo4j is running: `docker ps | grep neo4j`
- Verify credentials in `.env`
- Try Neo4j Browser: http://localhost:7474

**Missing spaCy model:**

**Using uv:**
```bash
uv run python -m spacy download en_core_web_sm
```

**Using Poetry:**
```bash
poetry run python -m spacy download en_core_web_sm
```

**Import errors:**
- Ensure you're in the project root
- Run `uv sync --all-extras` or `poetry install` again

## Next Steps

- Read the [Development Guide](DEVELOPMENT.md) for detailed setup
- Check out the [API Documentation](http://127.0.0.1:8000/docs)
- Explore the knowledge graph features
- Try uploading different types of documents

Happy exploring! 🚀
