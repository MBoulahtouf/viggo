# SOLID Refactoring Guide for Viggo Services

## Overview

This document describes the refactoring of the Viggo services directory to follow SOLID principles and improve the overall architecture of the RAG system.

## Problems with the Original Architecture

### 1. Single Responsibility Principle (SRP) Violations
- **RAGService** handled too many responsibilities: document processing, chunking, indexing, querying, and LLM generation
- **HybridRetriever** mixed retrieval logic with caching and performance optimization
- **GraphService** combined data storage with business logic

### 2. Open/Closed Principle (OCP) Violations
- Hard-coded dependencies made extension difficult
- No clear interfaces for adding new document processors or chunking strategies
- Tight coupling between components prevented easy modification

### 3. Liskov Substitution Principle (LSP) Violations
- No clear inheritance hierarchies or contracts
- Inconsistent interfaces between similar services

### 4. Interface Segregation Principle (ISP) Violations
- Services depended on large interfaces they didn't use
- No clear separation between different types of operations

### 5. Dependency Inversion Principle (DIP) Violations
- High-level modules depended directly on low-level modules
- No dependency injection or abstraction layers

## New Architecture

### Core Principles

1. **Separation of Concerns**: Each service has a single, well-defined responsibility
2. **Dependency Injection**: Services depend on abstractions, not concrete implementations
3. **Interface Segregation**: Small, focused interfaces for specific operations
4. **Open/Closed**: Easy to extend without modifying existing code
5. **Composition over Inheritance**: Services are composed of smaller, focused components

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (API Endpoints, User Interfaces, Business Logic)          │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  (RAGService, Orchestration, Coordination)                 │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   Interface Layer                           │
│  (Abstract Base Classes, Contracts, Protocols)             │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                 Implementation Layer                        │
│  (Concrete Classes, Specific Technologies)                 │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│  (FAISS, Neo4j, Redis, File System)                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Interfaces (`/interfaces/`)

#### Document Processing
- `DocumentProcessor`: Abstract base for document processors
- `DocumentProcessorFactory`: Factory for creating processors

#### Chunking
- `ChunkingStrategy`: Abstract base for chunking strategies
- `ChunkingService`: Service for managing chunking operations

#### Retrieval
- `Retriever`: Abstract base for retrieval operations
- `HybridRetriever`: Interface for combining multiple retrievers
- `ResultRanker`: Interface for ranking and reordering results

#### Generation
- `TextGenerator`: Abstract base for text generation
- `GenerationService`: Service for managing generation operations

#### Storage
- `StorageBackend`: Generic storage interface
- `VectorStorage`: Specialized interface for vector storage
- `GraphStorage`: Specialized interface for graph storage
- `CacheStorage`: Specialized interface for cache storage

#### RAG
- `RAGService`: Main RAG service interface
- `RAGOrchestrator`: Interface for creating and configuring RAG services

### 2. Implementations (`/implementations/`)

Concrete implementations of all interfaces, following the same structure as the interfaces directory.

### 3. Factory Service (`rag_factory.py`)

Provides easy access to the new architecture with:
- `RAGFactory`: Main factory class
- `get_rag_service()`: Convenience function for creating RAG services
- `get_legacy_compatible_service()`: Bridge to legacy code

## Migration Guide

### For New Code

Use the new architecture:

```python
from viggo.core.services import get_rag_service, GraphService, RedisService

# Create services
graph_service = GraphService(uri, user, password)
redis_service = RedisService()

# Create RAG service with new architecture
rag_service = get_rag_service(
    graph_service=graph_service,
    redis_service=redis_service,
    config_type="default"
)

# Use the service
result = rag_service.query("What is the story about?")
print(result.answer)
```

### For Legacy Code

Use the legacy-compatible wrapper:

```python
from viggo.core.services import get_legacy_compatible_service, GraphService

# Create legacy-compatible service
graph_service = GraphService(uri, user, password)
rag_service = get_legacy_compatible_service(graph_service=graph_service)

# Use existing interface
result = rag_service.perform_rag_query("What is the story about?")
print(result["answer"])
```

### For Custom Configurations

```python
from viggo.core.services import RAGFactory, LLMTextGenerator, SemanticRetriever

# Create custom configuration
factory = RAGFactory()
components = {
    'generators': [LLMTextGenerator()],
    'retrievers': [SemanticRetriever(vector_index, chunks_metadata)]
}

rag_service = factory.create_rag_service(
    config_type="custom",
    components=components
)
```

## Benefits of the New Architecture

### 1. Maintainability
- Clear separation of concerns
- Easy to understand and modify individual components
- Reduced coupling between services

### 2. Testability
- Each component can be tested in isolation
- Easy to mock dependencies
- Clear interfaces for unit testing

### 3. Extensibility
- Easy to add new document processors
- Simple to implement new chunking strategies
- Straightforward to add new retrieval sources

### 4. Flexibility
- Can mix and match different implementations
- Easy to swap out storage backends
- Configurable service composition

### 5. Performance
- Better resource management
- Optimized component interactions
- Improved caching strategies

## Configuration Options

### Default Configuration
- Uses all available services
- Optimized for production use
- Includes caching and performance optimization

### Minimal Configuration
- Basic functionality only
- Good for testing and development
- Minimal dependencies

### Custom Configuration
- Specify exact components to use
- Fine-grained control over behavior
- Optimized for specific use cases

## Best Practices

### 1. Use Interfaces
Always depend on interfaces, not concrete implementations:

```python
# Good
def process_document(processor: DocumentProcessor):
    return processor.process_document(file_path)

# Bad
def process_document(processor: PDFDocumentProcessor):
    return processor.process_document(file_path)
```

### 2. Dependency Injection
Inject dependencies rather than creating them:

```python
# Good
class RAGService:
    def __init__(self, retriever: HybridRetriever, generator: GenerationService):
        self.retriever = retriever
        self.generator = generator

# Bad
class RAGService:
    def __init__(self):
        self.retriever = ConcreteHybridRetriever()
        self.generator = ConcreteGenerationService()
```

### 3. Single Responsibility
Each class should have one reason to change:

```python
# Good - separate concerns
class DocumentProcessor:
    def process_document(self, file_path: str) -> List[DocumentPage]:
        pass

class ChunkingService:
    def chunk_document(self, pages: List[DocumentPage]) -> ChunkingResult:
        pass

# Bad - mixed responsibilities
class DocumentProcessor:
    def process_document(self, file_path: str) -> ChunkingResult:
        # Processing and chunking mixed together
        pass
```

### 4. Interface Segregation
Keep interfaces small and focused:

```python
# Good - focused interface
class VectorStorage:
    def add_vectors(self, vectors: List[List[float]]) -> bool:
        pass
    def search_vectors(self, query: List[float], top_k: int) -> List[Dict]:
        pass

# Bad - too many responsibilities
class StorageBackend:
    def add_vectors(self, vectors: List[List[float]]) -> bool:
        pass
    def add_nodes(self, nodes: List[Dict]) -> bool:
        pass
    def cache_data(self, key: str, data: Any) -> bool:
        pass
```

## Future Enhancements

### 1. Plugin System
- Dynamic loading of new components
- Runtime configuration changes
- Hot-swapping of implementations

### 2. Advanced Caching
- Multi-level caching strategies
- Intelligent cache invalidation
- Performance-based cache optimization

### 3. Monitoring and Metrics
- Built-in performance monitoring
- Health checks for all components
- Detailed metrics and analytics

### 4. Configuration Management
- External configuration files
- Environment-based settings
- Runtime configuration updates

## Conclusion

The new SOLID-compliant architecture provides a solid foundation for the Viggo RAG system. It addresses the original design issues while maintaining backward compatibility and providing a clear path for future enhancements.

The architecture is designed to be:
- **Maintainable**: Easy to understand and modify
- **Testable**: Clear interfaces and dependency injection
- **Extensible**: Simple to add new functionality
- **Flexible**: Configurable and composable
- **Performant**: Optimized for production use

By following the principles outlined in this guide, developers can build robust, maintainable, and extensible RAG services that will serve the project well into the future.
