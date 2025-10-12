# Development Guide

This guide provides comprehensive information for developers working on Viggo.

## Table of Contents

- [Local Setup](#local-setup)
- [Running the Application](#running-the-application)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Troubleshooting](#troubleshooting)
- [Architecture Overview](#architecture-overview)
- [Contributing](#contributing)

## Local Setup

### Prerequisites

- Python 3.11 or 3.12
- Poetry or uv (for dependency management - uv recommended for speed)
- Neo4j database (local or remote)
- Groq API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd viggo
   ```

2. **Install dependencies:**

   **Option A: Using uv (Recommended - Faster)**
   ```bash
   uv sync --all-extras
   ```

   **Option B: Using Poetry**
   ```bash
   poetry install
   ```

3. **Download spaCy model:**

   **Using uv:**
   ```bash
   uv run python -m spacy download en_core_web_sm
   ```

   **Using Poetry:**
   ```bash
   poetry run python -m spacy download en_core_web_sm
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.sample .env
   # Edit .env with your actual values
   ```

5. **Start Neo4j (if running locally):**
   ```bash
   # Using Docker
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:latest
   ```

### Environment Variables

Create a `.env` file with the following variables:

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

## Running the Application

### Development Server

**Using uv:**
```bash
uv run uvicorn viggo.main:app --reload
```

**Using Poetry:**
```bash
poetry run uvicorn viggo.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### API Documentation

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Basic Usage

1. **Upload a document:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_document.pdf"
   ```

2. **Query the document:**
   ```bash
   curl -X POST "http://127.0.0.1:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main character?", "page_number": 50}'
   ```

3. **Explore the knowledge graph:**
   ```bash
   curl -X GET "http://127.0.0.1:8000/api/v1/graph/nodes?label=Character"
   ```

## Testing

### Running Tests

**Using uv:**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=viggo --cov-report=html

# Run specific test file
uv run pytest tests/test_entity_utils.py

# Run with verbose output
uv run pytest -v
```

**Using Poetry:**
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=viggo --cov-report=html

# Run specific test file
poetry run pytest tests/test_entity_utils.py

# Run with verbose output
poetry run pytest -v
```

### Test Structure

- `tests/test_entity_utils.py` - Entity utility functions
- `tests/test_aliasing_service.py` - Aliasing service functionality
- `tests/test_graph_service.py` - Graph service with mocked Neo4j
- `tests/test_rag_service.py` - RAG service with stubbed models
- `tests/test_api.py` - API endpoint integration tests

### Writing Tests

Follow these guidelines:

1. **Use descriptive test names:**
   ```python
   def test_normalize_entity_name_with_whitespace():
       """Test entity name normalization with various whitespace patterns."""
   ```

2. **Mock external dependencies:**
   ```python
   @patch('viggo.core.services.rag_service.Groq')
   def test_llm_generation(self, mock_groq):
       # Test implementation
   ```

3. **Test both success and failure cases:**
   ```python
   def test_success_case():
       # Test happy path
   
   def test_failure_case():
       # Test error handling
   ```

## Dependency Management

Viggo uses a hybrid Poetry + uv approach for optimal performance and reproducibility:

### Poetry vs uv Usage

- **Poetry**: Used for lock file management and reproducibility
- **uv**: Used for fast dependency installation (CI and local development)

### Workflow

1. **Adding Dependencies:**
   - Edit `pyproject.toml` directly
   - Run `poetry lock` to update lock file
   - Use `uv sync` for fast installation

2. **Updating Dependencies:**
   ```bash
   # Update lock file with Poetry
   poetry lock --no-update
   
   # Install with uv for speed
   uv sync --all-extras
   ```

3. **Local Development:**
   - Use `uv run <command>` for faster execution
   - Use `poetry run <command>` if you prefer Poetry
   - Both respect the same lock file

### Benefits of Hybrid Approach

- **Speed**: uv is 2-3x faster than Poetry for dependency resolution
- **Reproducibility**: Poetry lock file ensures consistent builds
- **Compatibility**: Works with existing Poetry workflows
- **CI Performance**: Faster CI builds with uv

## Code Quality

### Linting and Formatting

**Using uv:**
```bash
# Check code style
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Type checking
uv run mypy viggo/ --ignore-missing-imports
```

**Using Poetry:**
```bash
# Check code style
poetry run ruff check .

# Fix auto-fixable issues
poetry run ruff check . --fix

# Format code
poetry run ruff format .

# Type checking
poetry run mypy viggo/ --ignore-missing-imports
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
poetry add -D pre-commit

# Install hooks
poetry run pre-commit install

# Run on all files
poetry run pre-commit run --all-files
```

### Code Style Guidelines

1. **Follow PEP 8** with line length of 88 characters
2. **Use type hints** for function parameters and return values
3. **Write docstrings** for all public functions and classes
4. **Use descriptive variable names**
5. **Keep functions small and focused**

## Troubleshooting

### Common Issues

#### 1. Neo4j Connection Errors

**Error:** `Failed to connect to Neo4j`

**Solutions:**
- Verify Neo4j is running: `docker ps | grep neo4j`
- Check connection details in `.env`
- Ensure firewall allows port 7687
- Try connecting with Neo4j Browser: `http://localhost:7474`

#### 2. spaCy Model Not Found

**Error:** `OSError: [E050] Can't find model 'en_core_web_sm'`

**Solution:**

**Using uv:**
```bash
uv run python -m spacy download en_core_web_sm
```

**Using Poetry:**
```bash
poetry run python -m spacy download en_core_web_sm
```

#### 3. Groq API Errors

**Error:** `API key not found` or `Rate limit exceeded`

**Solutions:**
- Verify `GROQ_API_KEY` in `.env`
- Check API key validity at Groq console
- Implement rate limiting for production use

#### 4. Memory Issues with Large Documents

**Error:** `Out of memory` during PDF processing

**Solutions:**
- Process documents in smaller chunks
- Increase system memory
- Use streaming processing for very large files

#### 5. Import Errors

**Error:** `ModuleNotFoundError: No module named 'viggo'`

**Solutions:**
- Ensure you're in the project root directory
- Run `uv sync --all-extras` or `poetry install` to install dependencies
- Check Python path configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor performance with:

```bash
# Install profiling tools
poetry add -D py-spy memory-profiler

# Profile CPU usage
poetry run py-spy top --pid <process_id>

# Profile memory usage
poetry run python -m memory_profiler your_script.py
```

## Architecture Overview

### Project Structure

```
viggo/
├── viggo/
│   ├── api/                 # FastAPI routes
│   │   └── v1/
│   │       └── endpoints/   # API endpoints
│   ├── core/               # Core business logic
│   │   ├── services/       # Service layer
│   │   ├── utils/          # Utility functions
│   │   └── config.py       # Configuration
│   ├── models/             # Pydantic models
│   └── main.py            # FastAPI app
├── tests/                  # Test files
├── docs/                   # Documentation
└── .github/workflows/      # CI/CD
```

### Key Components

1. **RAGService** - Handles document processing and retrieval
2. **GraphService** - Manages Neo4j knowledge graph
3. **AliasingService** - Handles entity aliases and synonyms
4. **Entity Utils** - Entity normalization and mapping

### Data Flow

1. **Document Upload** → PDF processing → Chunking → Vector indexing
2. **Entity Extraction** → spaCy NER → Graph loading
3. **Query Processing** → Vector search → LLM generation
4. **Graph Queries** → Neo4j traversal → Entity relationships

## Contributing

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes and test:**

   **Using uv:**
   ```bash
   uv run pytest
   uv run ruff check .
   ```

   **Using Poetry:**
   ```bash
   poetry run pytest
   poetry run ruff check .
   ```

3. **Commit with descriptive messages:**
   ```bash
   git commit -m "feat: add entity aliasing functionality"
   ```

4. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements

### Pull Request Guidelines

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Add tests** for new functionality
4. **Follow code style** guidelines
5. **Provide clear description** of changes

### Code Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are comprehensive
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/)
- [spaCy Documentation](https://spacy.io/usage)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [pytest Documentation](https://docs.pytest.org/)

For questions or issues, please create an issue in the repository or contact the development team.
