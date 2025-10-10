# Elasticsearch Migration Plan

## Current FAISS Usage Analysis

### What FAISS Currently Does:
1. **Vector Storage**: 384-dimensional embeddings (all-MiniLM-L6-v2)
2. **Similarity Search**: L2 distance with top-k retrieval
3. **Metadata Storage**: Content, entities, page numbers, chunk IDs
4. **Index Persistence**: Save/load from disk

### Migration to Elasticsearch

## 1. Index Schema Design

```json
{
  "mappings": {
    "properties": {
      "content": {
        "type": "text",
        "analyzer": "standard"
      },
      "content_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "l2_norm"
      },
      "page": {
        "type": "integer"
      },
      "chunk_id": {
        "type": "keyword"
      },
      "entities": {
        "type": "keyword"
      },
      "entity_labels": {
        "type": "keyword"
      },
      "chapter_title": {
        "type": "text"
      },
      "chunk_type": {
        "type": "keyword"
      },
      "lore_significance": {
        "type": "float"
      },
      "word_count": {
        "type": "integer"
      },
      "char_count": {
        "type": "integer"
      }
    }
  }
}
```

## 2. ElasticsearchVectorStorage Implementation

```python
class ElasticsearchVectorStorage(VectorStorage):
    def __init__(self, es_client, index_name: str = "viggo-vectors"):
        self.es_client = es_client
        self.index_name = index_name
        self.dimension = 384
        
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, Any]]) -> bool:
        """Add vectors to Elasticsearch index."""
        try:
            bulk_actions = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                doc = {
                    "content_vector": vector,
                    **meta
                }
                bulk_actions.append({
                    "_index": self.index_name,
                    "_source": doc
                })
            
            # Bulk index
            response = helpers.bulk(self.es_client, bulk_actions)
            return response[0] > 0
            
        except Exception as e:
            print(f"Error adding vectors to Elasticsearch: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search for similar vectors using Elasticsearch."""
        try:
            query = {
                "knn": {
                    "field": "content_vector",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": top_k * 10
                }
            }
            
            response = self.es_client.search(
                index=self.index_name,
                body={"query": query},
                size=top_k
            )
            
            results = []
            for hit in response['hits']['hits']:
                result = {
                    "content": hit['_source'].get('content', ''),
                    "score": hit['_score'],
                    "metadata": hit['_source'],
                    "index": hit['_id']
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors in Elasticsearch: {e}")
            return []
```

## 3. Hybrid Search Implementation

```python
class ElasticsearchHybridRetriever:
    def __init__(self, es_client, index_name: str = "viggo-vectors"):
        self.es_client = es_client
        self.index_name = index_name
        
    def hybrid_search(self, query: str, top_k: int = 5, page_filter: Optional[int] = None):
        """Perform hybrid search combining vector and keyword search."""
        
        # Generate query embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode(query).tolist()
        
        # Build hybrid query
        query_body = {
            "query": {
                "bool": {
                    "should": [
                        # Vector similarity search
                        {
                            "knn": {
                                "field": "content_vector",
                                "query_vector": query_vector,
                                "k": top_k,
                                "num_candidates": top_k * 10,
                                "boost": 0.7
                            }
                        },
                        # Keyword search
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["content^2", "chapter_title^1.5", "entities"],
                                "type": "best_fields",
                                "boost": 0.3
                            }
                        }
                    ]
                }
            }
        }
        
        # Add page filter if specified
        if page_filter is not None:
            query_body["query"]["bool"]["filter"] = [
                {"range": {"page": {"lte": page_filter}}}
            ]
        
        response = self.es_client.search(
            index=self.index_name,
            body=query_body,
            size=top_k
        )
        
        return self._format_results(response)
```

## 4. Migration Steps

### Phase 1: Setup Elasticsearch
1. Install and configure Elasticsearch with vector search support
2. Create index with proper mapping
3. Implement ElasticsearchVectorStorage class

### Phase 2: Parallel Implementation
1. Run both FAISS and Elasticsearch in parallel
2. Compare results and performance
3. Validate data consistency

### Phase 3: Switch Over
1. Update configuration to use Elasticsearch
2. Remove FAISS dependencies
3. Update tests and documentation

## 5. Performance Considerations

### Elasticsearch Configuration:
```yaml
# elasticsearch.yml
indices.memory.index_buffer_size: 30%
indices.queries.cache.size: 20%
indices.fielddata.cache.size: 20%

# For vector search optimization
index.knn.algo_param.ef_search: 100
index.knn.algo_param.ef_construction: 200
```

### Index Settings:
```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.knn": true,
    "index.knn.algo_param.ef_search": 100
  }
}
```

## 6. Benefits of Migration

1. **Unified Search**: Single platform for all search operations
2. **Better Filtering**: Rich metadata filtering capabilities
3. **Scalability**: Built-in clustering and horizontal scaling
4. **Monitoring**: Better observability with Kibana
5. **Maintenance**: Easier to manage one search engine
6. **Hybrid Search**: Native combination of vector and keyword search

## 7. Trade-offs

1. **Performance**: Slightly slower than FAISS for pure vector operations
2. **Memory**: Higher memory usage
3. **Complexity**: More complex setup and configuration
4. **Dependencies**: Additional Elasticsearch infrastructure

## Conclusion

The migration is feasible and recommended for your use case because:
- You're already using Azure Cognitive Search (similar to Elasticsearch)
- Your dataset size is manageable for Elasticsearch
- You benefit from unified search capabilities
- The performance difference is negligible for your scale
