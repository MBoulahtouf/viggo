# Elasticsearch Migration Guide

This guide explains how to migrate from FAISS to Elasticsearch for the Viggo RAG system.

## Overview

The migration replaces FAISS vector storage with Elasticsearch, providing:
- Unified search platform (vector + keyword + structured search)
- Better scalability and distributed search
- Rich filtering and querying capabilities
- Improved monitoring and management tools

## Prerequisites

1. **Docker and Docker Compose** (for easy Elasticsearch setup)
2. **Python 3.12+** with the updated dependencies

## Migration Steps

### 1. Install Dependencies

Update your dependencies by running:
```bash
poetry install
```

This will install `elasticsearch` instead of `faiss-cpu`.

### 2. Start Elasticsearch

Use the provided Docker Compose file:
```bash
docker-compose -f docker-compose.elasticsearch.yml up -d
```

This will start:
- Elasticsearch on `localhost:9200`
- Kibana on `localhost:5601` (optional, for monitoring)

### 3. Verify Elasticsearch is Running

Check that Elasticsearch is accessible:
```bash
curl http://localhost:9200
```

You should see a JSON response with cluster information.

### 4. Run Migration Script

Execute the migration script to test the setup:
```bash
python migrate_to_elasticsearch.py
```

This script will:
- Check Elasticsearch connectivity
- Create the vector index with proper mapping
- Migrate sample data
- Test search functionality

### 5. Update Your Application

The following changes have been made to your codebase:

#### Storage Implementation
- `FAISSVectorStorage` → `ElasticsearchVectorStorage`
- Updated imports in `viggo/core/services/implementations/__init__.py`

#### Retrievers
- `SemanticRetriever` now uses `ElasticsearchVectorStorage`
- `HybridRetriever` updated to work with Elasticsearch

#### Configuration
- Added Elasticsearch settings to `viggo/core/config.py`
- New environment variables for local Elasticsearch

### 6. Test the Migration

Run the test suite to ensure everything works:
```bash
python -m pytest tests/test_solid_architecture.py::test_vector_storage -v
```

## Configuration

### Environment Variables

Add these to your `.env` file:
```env
# Local Elasticsearch Configuration
LOCAL_ELASTICSEARCH_HOST=localhost
LOCAL_ELASTICSEARCH_PORT=9200
LOCAL_ELASTICSEARCH_INDEX=viggo-vectors
```

### Elasticsearch Settings

The system creates an index with the following configuration:
- **Vector Field**: 384-dimensional dense vectors with L2 similarity
- **Text Fields**: Content, chapter titles, entities
- **Metadata Fields**: Page numbers, chunk types, lore significance
- **Performance**: Optimized for vector search with HNSW algorithm

## Index Schema

```json
{
  "mappings": {
    "properties": {
      "content": {"type": "text", "analyzer": "standard"},
      "content_vector": {
        "type": "dense_vector",
        "dims": 384,
        "index": true,
        "similarity": "l2_norm"
      },
      "page": {"type": "integer"},
      "chunk_id": {"type": "keyword"},
      "entities": {"type": "keyword"},
      "entity_labels": {"type": "keyword"},
      "chapter_title": {"type": "text"},
      "chunk_type": {"type": "keyword"},
      "lore_significance": {"type": "float"},
      "word_count": {"type": "integer"},
      "char_count": {"type": "integer"}
    }
  }
}
```

## Performance Considerations

### Elasticsearch Configuration
- **Memory**: Allocated 1GB heap for development
- **Index Settings**: Single shard, no replicas for development
- **Vector Search**: HNSW algorithm with ef_search=100

### Production Recommendations
- Increase heap size based on data volume
- Add replicas for high availability
- Tune ef_search parameter for better recall/precision trade-off
- Consider multiple shards for large datasets

## Monitoring

### Kibana Dashboard
Access Kibana at `http://localhost:5601` to:
- Monitor cluster health
- View index statistics
- Analyze search performance
- Debug queries

### Health Checks
```bash
# Cluster health
curl http://localhost:9200/_cluster/health

# Index statistics
curl http://localhost:9200/viggo-vectors/_stats

# Search test
curl -X POST http://localhost:9200/viggo-vectors/_search \
  -H "Content-Type: application/json" \
  -d '{"query": {"match_all": {}}, "size": 1}'
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure Elasticsearch is running: `docker ps`
   - Check port 9200 is accessible: `curl http://localhost:9200`

2. **Index Creation Failed**
   - Check Elasticsearch logs: `docker logs viggo-elasticsearch`
   - Verify mapping syntax is correct

3. **Search Returns No Results**
   - Check if data was indexed: `curl http://localhost:9200/viggo-vectors/_count`
   - Verify vector dimensions match (384 for all-MiniLM-L6-v2)

4. **Performance Issues**
   - Increase heap size in docker-compose.yml
   - Tune ef_search parameter
   - Consider using multiple shards

### Logs
```bash
# Elasticsearch logs
docker logs viggo-elasticsearch

# Application logs
# Check your application logs for Elasticsearch-related errors
```

## Rollback Plan

If you need to rollback to FAISS:

1. **Revert Code Changes**
   ```bash
   git checkout HEAD~1 -- viggo/core/services/implementations/storage_impl.py
   git checkout HEAD~1 -- viggo/core/services/implementations/retrieval_impl.py
   git checkout HEAD~1 -- viggo/core/services/implementations/hybrid_retriever_impl.py
   ```

2. **Restore Dependencies**
   ```bash
   # In pyproject.toml, change:
   # "elasticsearch (>=8.0.0,<9.0.0)" back to "faiss-cpu (>=1.11.0,<2.0.0)"
   poetry install
   ```

3. **Restart Application**
   ```bash
   # Stop Elasticsearch
   docker-compose -f docker-compose.elasticsearch.yml down
   
   # Restart your application
   ```

## Benefits of Migration

1. **Unified Search**: Single platform for vector, keyword, and structured search
2. **Better Filtering**: Rich metadata filtering capabilities
3. **Scalability**: Built-in clustering and horizontal scaling
4. **Monitoring**: Better observability with Kibana
5. **Maintenance**: Easier to manage one search engine
6. **Hybrid Search**: Native combination of vector and keyword search

## Next Steps

After successful migration:

1. **Monitor Performance**: Use Kibana to track search performance
2. **Optimize Settings**: Tune Elasticsearch parameters based on usage
3. **Scale as Needed**: Add more nodes for production workloads
4. **Backup Strategy**: Implement regular index snapshots
5. **Security**: Enable authentication and encryption for production

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Elasticsearch documentation
3. Check application logs for specific error messages
4. Test with the migration script to isolate issues
