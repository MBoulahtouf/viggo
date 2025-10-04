# Viggo Hybrid Chunking Strategy Guide

## Overview

The Hybrid Chunking Strategy is designed to reduce noise and improve retrieval accuracy in Viggo's book lore knowledge explorer. It implements a hierarchical approach that combines pre-chunking, post-chunking, and overlapping chunks to provide optimal context for different types of queries.

## Architecture

### 1. Hierarchical Chunking Levels

```
Book
├── Chapter (Pre-chunking)
│   ├── Section (if chapter is too large)
│   └── Passage (Post-chunking)
│       ├── Standard Passage
│       ├── Dialogue Block
│       ├── Narrative Block
│       └── Critical Lore (Overlapping)
└── Sentence (Overlapping chunks for critical lore)
```

### 2. Chunk Types

- **Full Chapter**: Complete chapters for broad context
- **Paragraph Group**: Grouped paragraphs for medium context
- **Standard Passage**: Regular passages for detailed answers
- **Overlapping Passage**: Critical lore with extended context
- **Dialogue Block**: Dialogue-focused chunks
- **Narrative Block**: Narrative-focused chunks
- **Critical Lore**: High-significance content with overlap

## Configuration

### ChunkingConfig Parameters

```python
from viggo.core.services.hybrid_chunking_service import ChunkingConfig

config = ChunkingConfig(
    # Pre-chunking settings
    max_chapter_words=2000,      # Maximum words per chapter
    min_chapter_words=100,       # Minimum words per chapter
    
    # Post-chunking settings
    max_passage_words=400,       # Maximum words per passage
    min_passage_words=50,        # Minimum words per passage
    passage_overlap_ratio=0.2,   # 20% overlap between passages
    
    # Overlapping chunk settings
    critical_lore_threshold=0.7, # Threshold for critical lore detection
    max_overlap_chunks=3,        # Maximum overlapping chunks to create
    
    # Hierarchical settings
    enable_hierarchical=True,    # Enable hierarchical relationships
    max_children_per_parent=10,  # Maximum children per parent chunk
    
    # Content filtering
    enable_content_filtering=True, # Enable content filtering
    skip_metadata_pages=2,       # Skip first N pages (often metadata)
)
```

### Recommended Configurations

#### For Academic/Literary Texts
```python
academic_config = ChunkingConfig(
    max_chapter_words=3000,
    max_passage_words=500,
    passage_overlap_ratio=0.25,
    critical_lore_threshold=0.6,
    enable_content_filtering=True,
    skip_metadata_pages=3
)
```

#### For Fiction/Novels
```python
fiction_config = ChunkingConfig(
    max_chapter_words=1500,
    max_passage_words=300,
    passage_overlap_ratio=0.2,
    critical_lore_threshold=0.7,
    enable_content_filtering=True,
    skip_metadata_pages=2
)
```

#### For Technical Documents
```python
technical_config = ChunkingConfig(
    max_chapter_words=1000,
    max_passage_words=200,
    passage_overlap_ratio=0.3,
    critical_lore_threshold=0.8,
    enable_content_filtering=True,
    skip_metadata_pages=1
)
```

## Usage Examples

### 1. Basic Hybrid Chunking

```python
from viggo.core.services.rag_service import RAGService
from viggo.core.services.graph_service import GraphService

# Initialize services
graph_service = GraphService()
rag_service = RAGService(graph_service)

# Process document with hybrid chunking
result = rag_service.process_document_hybrid_chunking("path/to/document.pdf")

print(f"Total chunks: {result['num_chunks']}")
print(f"Processing method: {result['processing_method']}")
```

### 2. Enhanced RAG Service

```python
from viggo.core.services.enhanced_rag_service import EnhancedRAGService
from viggo.core.services.hybrid_chunking_service import ChunkLevel

# Initialize enhanced RAG service
enhanced_rag = EnhancedRAGService(graph_service)

# Process document
result = enhanced_rag.process_document_enhanced("path/to/document.pdf")

# Query at different hierarchical levels
chapter_result = enhanced_rag.query_hierarchical(
    "Tell me about the overall story", 
    ChunkLevel.CHAPTER, 
    top_k=3
)

passage_result = enhanced_rag.query_hierarchical(
    "What is the strange high house?", 
    ChunkLevel.PASSAGE, 
    top_k=5
)

sentence_result = enhanced_rag.query_hierarchical(
    "What exactly does Olney see?", 
    ChunkLevel.SENTENCE, 
    top_k=3
)
```

### 3. Custom Chunking Configuration

```python
from viggo.core.services.hybrid_chunking_service import ChunkingConfig

# Create custom configuration
custom_config = ChunkingConfig(
    max_chapter_words=1800,
    max_passage_words=350,
    passage_overlap_ratio=0.22,
    critical_lore_threshold=0.65,
    max_overlap_chunks=5
)

# Apply to RAG service
rag_service.chunking_config = custom_config
rag_service.hybrid_chunking = HybridChunkingService(config=custom_config)

# Process with custom configuration
result = rag_service.process_document_hybrid_chunking("document.pdf")
```

## Query Strategies

### 1. Level Selection Based on Query Type

```python
def determine_query_level(query: str) -> ChunkLevel:
    """Determine optimal chunk level based on query characteristics."""
    query_lower = query.lower()
    
    # Broad context queries → Chapter level
    if any(word in query_lower for word in ['overview', 'summary', 'tell me about', 'explain']):
        return ChunkLevel.CHAPTER
    
    # Specific entity queries → Passage level
    elif any(word in query_lower for word in ['who is', 'what is', 'where is', 'when did']):
        return ChunkLevel.PASSAGE
    
    # Detailed analysis queries → Sentence level
    elif any(word in query_lower for word in ['exactly', 'precisely', 'specifically', 'in detail']):
        return ChunkLevel.SENTENCE
    
    # Default to passage level
    else:
        return ChunkLevel.PASSAGE
```

### 2. Hierarchical Query Expansion

```python
def hierarchical_query_expansion(rag_service, query: str, top_k: int = 5):
    """Perform hierarchical query with automatic level selection and expansion."""
    
    # Determine optimal level
    optimal_level = determine_query_level(query)
    
    # Search at optimal level
    results = rag_service.query_hierarchical(query, optimal_level, top_k)
    
    # If insufficient results, expand to other levels
    if len(results['results']) < top_k:
        # Try broader context
        if optimal_level != ChunkLevel.CHAPTER:
            broader_results = rag_service.query_hierarchical(
                query, ChunkLevel.CHAPTER, top_k - len(results['results'])
            )
            results['results'].extend(broader_results['results'])
        
        # Try more specific context
        if optimal_level != ChunkLevel.SENTENCE:
            specific_results = rag_service.query_hierarchical(
                query, ChunkLevel.SENTENCE, top_k - len(results['results'])
            )
            results['results'].extend(specific_results['results'])
    
    return results
```

## Performance Optimization

### 1. Chunk Size Optimization

```python
# Monitor chunk statistics
stats = rag_service.get_chunking_statistics()

# Adjust based on performance
if stats['avg_words_per_chunk'] > 500:
    # Chunks too large, reduce max_passage_words
    config.max_passage_words = 300
elif stats['avg_words_per_chunk'] < 100:
    # Chunks too small, increase max_passage_words
    config.max_passage_words = 500
```

### 2. Lore Significance Tuning

```python
# Analyze critical lore chunks
critical_chunks = rag_service.get_critical_lore_chunks(threshold=0.7)

# Adjust threshold based on results
if len(critical_chunks) > 50:
    # Too many critical chunks, increase threshold
    config.critical_lore_threshold = 0.8
elif len(critical_chunks) < 10:
    # Too few critical chunks, decrease threshold
    config.critical_lore_threshold = 0.6
```

### 3. Overlap Ratio Optimization

```python
# Test different overlap ratios
overlap_ratios = [0.1, 0.2, 0.3, 0.4]
best_ratio = 0.2
best_score = 0

for ratio in overlap_ratios:
    config.passage_overlap_ratio = ratio
    # Process document and test retrieval quality
    score = test_retrieval_quality(rag_service, test_queries)
    
    if score > best_score:
        best_score = score
        best_ratio = ratio

config.passage_overlap_ratio = best_ratio
```

## Monitoring and Analytics

### 1. Chunking Statistics

```python
# Get comprehensive statistics
stats = rag_service.get_chunking_statistics()

print(f"Total chunks: {stats['total_chunks']}")
print(f"Chunks by level: {stats['chunks_by_level']}")
print(f"Chunks by type: {stats['chunks_by_type']}")
print(f"Word count stats: {stats['word_count_stats']}")
print(f"Lore significance stats: {stats['lore_significance_stats']}")
```

### 2. Retrieval Performance

```python
# Monitor retrieval performance
def analyze_retrieval_performance(rag_service, test_queries):
    performance_metrics = {
        'avg_response_time': 0,
        'avg_relevance_score': 0,
        'coverage_by_level': {},
        'critical_lore_usage': 0
    }
    
    for query in test_queries:
        start_time = time.time()
        result = rag_service.perform_rag_query(query)
        response_time = time.time() - start_time
        
        performance_metrics['avg_response_time'] += response_time
        # Add more metrics as needed
    
    return performance_metrics
```

### 3. Content Quality Metrics

```python
# Analyze content quality
def analyze_content_quality(chunks):
    quality_metrics = {
        'avg_entity_density': 0,
        'avg_lore_significance': 0,
        'noise_ratio': 0,
        'context_completeness': 0
    }
    
    for chunk in chunks:
        # Calculate various quality metrics
        entity_density = len(chunk.get('entities', [])) / chunk.get('word_count', 1)
        quality_metrics['avg_entity_density'] += entity_density
        
        lore_sig = chunk.get('lore_significance', 0)
        quality_metrics['avg_lore_significance'] += lore_sig
    
    # Normalize metrics
    for metric in quality_metrics:
        quality_metrics[metric] /= len(chunks)
    
    return quality_metrics
```

## Best Practices

### 1. Document Type Considerations

- **Fiction**: Use smaller chunks (300-400 words) with higher overlap (20-25%)
- **Academic**: Use larger chunks (400-500 words) with moderate overlap (15-20%)
- **Technical**: Use medium chunks (200-300 words) with high overlap (25-30%)

### 2. Query Pattern Optimization

- **Broad questions**: Use chapter-level retrieval
- **Specific questions**: Use passage-level retrieval
- **Detailed questions**: Use sentence-level retrieval with overlapping chunks

### 3. Memory and Storage Management

```python
# Clear old data before processing new documents
rag_service.clear_data()

# Use efficient storage for large documents
if document_size > 10MB:
    config.max_chapter_words = 1000
    config.max_passage_words = 200
```

### 4. Error Handling

```python
try:
    result = rag_service.process_document_hybrid_chunking(file_path)
    if result['num_chunks'] == 0:
        print("Warning: No chunks generated, check document format")
        # Fall back to traditional chunking
        result = rag_service.process_document_enhanced(file_path)
except Exception as e:
    print(f"Hybrid chunking failed: {e}")
    # Fall back to traditional processing
    result = rag_service.process_document(file_path)
```

## Troubleshooting

### Common Issues

1. **No chunks generated**: Check document format and content filtering settings
2. **Poor retrieval quality**: Adjust chunk sizes and overlap ratios
3. **Memory issues**: Reduce chunk sizes and enable content filtering
4. **Slow processing**: Optimize chunking configuration and use parallel processing

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug configuration
debug_config = ChunkingConfig(
    max_chapter_words=500,  # Smaller for debugging
    max_passage_words=100,
    enable_content_filtering=True,
    skip_metadata_pages=0  # Don't skip pages for debugging
)
```

## Integration with Existing Systems

### 1. Backward Compatibility

The hybrid chunking strategy maintains backward compatibility with existing RAG service methods:

```python
# Existing methods still work
result = rag_service.process_document(file_path)
result = rag_service.process_document_enhanced(file_path)

# New hybrid method
result = rag_service.process_document_hybrid_chunking(file_path)
```

### 2. Azure Cognitive Search Integration

```python
# Index hierarchical chunks in Azure Search
rag_service.index_document_in_azure_search(file_path)

# Query with hierarchical context
results = rag_service.hybrid_search_service.hybrid_search(query, k=5)
```

### 3. Neo4j Graph Integration

```python
# Build knowledge graph from hierarchical chunks
for chunk in result['chunks_with_metadata']:
    entities = chunk.get('entities', [])
    relationships = chunk.get('relationships', [])
    
    # Add to Neo4j graph
    graph_service.add_entities(entities)
    graph_service.add_relationships(relationships)
```

## Conclusion

The Hybrid Chunking Strategy provides a comprehensive solution for reducing noise and improving retrieval accuracy in book lore knowledge exploration. By implementing hierarchical chunking with pre-chunking, post-chunking, and overlapping strategies, Viggo can provide more accurate and contextually relevant answers to user queries.

Key benefits:
- **Reduced noise** through content filtering and entity enhancement
- **Improved context** through hierarchical retrieval
- **Better precision** through overlapping chunks for critical lore
- **Flexible configuration** for different document types
- **Backward compatibility** with existing systems

For more information, see the demo script (`demo_hybrid_chunking.py`) and the enhanced RAG service implementation.
